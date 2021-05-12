#!/usr/bin/env python
import argparse
import asyncio
import glob
import json
import logging
import os
import re
import time
import unicodedata
import sys
import mimetypes
import difflib
from datetime import datetime, timedelta
from statistics import mean

# ffmpeg has lots of output, not all of them actual failures, so ignore ones that we know are generally ok
# these are just warnings, should play ok, but there are minor errors
ffmpeg_warning_patterns = [
    r'.*Application provided invalid, non monotonically increasing dts to muxer in stream .*',
    r'.*\[mp3float\] Header missing\.*',
    r'.*Last message repeated \d* times.*'
]
# these are nothing to worry about, we can ignore them completely
ffmpeg_ignore_patterns = [
    r'.*Last message repeated \d* times.*'
]

# Certain codecs can use hardware acceleration to decode, this maps codecs to the proper ffmpeg acceleration params
# Use ffmpeg -decoders to see all available decoders
# For NVidia decoding: ffmpeg -decoders | grep cuvid
# Note: even if it is listed, your card may not support this
# Should probably auto-detect this, but don't know a good way
haccel_codecs = {
    # "hevc": ["-c:v", "hevc_cuvid"],
    # "vp8": ["-c:v", "vp8_cuvid"],
    # "vp9": ["-c:v", "vp9_cuvid"],
    # "vc1": ["-c:v", "vc1_cuvid"],
    # "av1": ["-c:v", "av1_cuvid"],
    "mpeg4": ["-c:v", "mpeg4_cuvid"],
    "h264": ["-c:v", "h264_cuvid"]
}

# Find the right ffmpeg/ffprobe paths
# Should probably move this to an option
os_paths = (os.getenv('PATH') or "").split(os.pathsep)
ffmpeg_paths = [
    *[os.path.join(p, 'ffmpeg') for p in os_paths],
    # Comment out the above to use a specific ffmpeg path like below
    # "/usr/bin/ffmpeg",
    # "/usr/local/bin/ffmpeg"
]
ffprobe_paths = [
    *[os.path.join(p, 'ffprobe') for p in os_paths],
    # Comment out the above to use a specific ffprobe path like below
    # "/usr/bin/ffprobe",
    # "/usr/local/bin/ffprobe"
]
FFMPEG_PATH = next((p for p in ffmpeg_paths if os.path.exists(p)), None)
FFPROBE_PATH = next((p for p in ffprobe_paths if os.path.exists(p)), None)
if FFMPEG_PATH is None:
    sys.exit('could not find ffmpeg')
if FFPROBE_PATH is None:
    sys.exit('could not find ffprobe')

mimetypes.init()


# Take a program + args and run it, combine stdout and stderr
async def run_command(command):
    # Create subprocess
    command_run_date = datetime.utcnow().isoformat()
    started_clock = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )

    # Status
    logging.debug("Started: %r, pid=%s" % (' '.join(command), process.pid))

    # Wait for the subprocess to finish
    stdout, stderr = await process.communicate()
    ended_clock = time.monotonic()

    output = stdout.decode().strip()
    # Progress
    if process.returncode == 0:
        logging.debug(
            "Done: %r, pid=%s, result: %s"
            % (' '.join(command), process.pid, output)
        )
    else:
        logging.debug(
            "Failed: %r, pid=%s, returncode=%s,result: %s"
            % (' '.join(command), process.pid, process.returncode, output)
        )

    stats = {
        "start_clock": started_clock,
        "end_clock": ended_clock,
        "date": command_run_date
    }

    return output, process.returncode == 0, stats


# ffprobe worker
async def video_probe_consumer_runner(task_list, completion_queue):
    while len(task_list) > 0:
        result_entry = task_list.pop()

        command = [FFPROBE_PATH, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of",
                   "default=noprint_wrappers=1:nokey=1", result_entry["filepath"]]

        result, successfull, stats = await run_command(command)

        result_entry["probed"] = True
        result_entry["probe_passed"] = successfull
        result_entry["probe_stats"] = stats

        if successfull:
            result_entry["codec"] = result
            result_entry["haccel_compatible"] = bool(haccel_codecs.get(result_entry["codec"], False))
        else:
            result_entry["raw_probe_results"] = result

        completion_queue.put_nowait(("probe", result_entry))


ffmpeg_output_cleaners = [
    r"frame=.+?\r",
    r"\[(.+?) @ 0x.+?\]", "[\\1]",
    r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]'
]
ffmpeg_output_cleaners_regex = [re.compile(regex) for regex in ffmpeg_output_cleaners]


def clean_ffmpeg_output(raw_results):
    cleaned_message = raw_results or ""
    for regex in ffmpeg_output_cleaners_regex:
        cleaned_message = regex.sub('', cleaned_message)

    lines = []
    for message in cleaned_message.splitlines():
        # remove any remaining control chars
        clean_line = "".join(ch for ch in message if unicodedata.category(ch)[0] != "C")
        if clean_line.startswith("'"):
            clean_line = clean_line[1:]
        if clean_line.endswith("'"):
            clean_line = clean_line[:-1]
        lines.append(clean_line)

    return lines


async def video_scan_consumer_runner(task_list, is_haccel_runner, completion_queue):
    while len(task_list) > 0:
        is_haccel_task = False
        if is_haccel_runner:
            # if haccel runner, try to find next haccel task
            task_index = next((index for (index, task) in enumerate(task_list) if
                               task["haccel_compatible"] and not task['force_cpu_only']), None)
            if task_index is None:
                # we have finished all haccel tasks
                break
            is_haccel_task = True
        else:
            # if not haccel runner and we have finished cpu tasks, start on haccel list
            task_index = next((index for (index, task) in enumerate(task_list) if not task["haccel_compatible"]),
                              False) or 0

        result_entry = task_list.pop(task_index)

        ha_flags = []
        if is_haccel_task:
            ha_flags.extend(haccel_codecs[result_entry["codec"]])

        queue_size_bug_flags = []
        if result_entry['retry_scan_queue_size_bug'] and not result_entry['force_cpu_only_no_queue_size']:
            queue_size_bug_flags = ["-max_muxing_queue_size", "9999"]

        command = [FFMPEG_PATH, "-v", "error", *ha_flags, "-i", result_entry["filepath"],
                   *queue_size_bug_flags, "-f", "null", "-"]
        result, successful, stats = await run_command(command)

        cleaned_result = clean_ffmpeg_output(result)

        too_many_packets_bug_detected = next(
            (r for r in cleaned_result if "Too many packets buffered for output stream" in r), False)
        if not result_entry['retry_scan_queue_size_bug'] and too_many_packets_bug_detected:
            # There is this ffmpeg weirdness with this error, if we encounter it, re-queue this scan with extra params
            result_entry['retry_scan_queue_size_bug'] = True
            logging.info(f"Retrying {os.path.basename(result_entry['filepath'])} with potential bug fix")
            task_list.append(result_entry)
        elif result_entry['retry_scan_queue_size_bug'] and too_many_packets_bug_detected and is_haccel_task:
            # If this bug STILL presents itself, may be a problem with haccel, so let's requeue with cpu-only
            result_entry['force_cpu_only'] = True
            logging.info(f"Retrying {os.path.basename(result_entry['filepath'])} with cpu only")
            task_list.append(result_entry)
        elif result_entry['retry_scan_queue_size_bug'] and too_many_packets_bug_detected and is_haccel_task:
            # Ok, last try, cpu only without queue params
            result_entry['force_cpu_only_no_queue_size'] = True
            logging.info(f"Retrying {os.path.basename(result_entry['filepath'])} with cpu only, no queue size fix")
            task_list.append(result_entry)
        else:
            result_entry["scanned"] = True
            result_entry["raw_scan_results"] = cleaned_result
            result_entry["scan_passed"] = successful
            result_entry["scan_stats"] = stats
            result_entry["scan_has_warnings"] = result != ""

            job_type = "scan:cpu"
            if is_haccel_task:
                job_type = "scan:haccel"

            completion_queue.put_nowait((job_type, result_entry))


def get_all_video_file_paths(folderpath):
    extensions = [ext[1:] for ext, mime in mimetypes.types_map.items() if mime.startswith('video')]
    paths = []
    for extension in extensions:
        paths.extend(glob.glob(os.path.join(folderpath, "*." + extension), recursive=True))
        paths.extend(glob.glob(os.path.join(folderpath, "**", "*." + extension), recursive=True))
    return set((os.path.abspath(path) for path in paths))


def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


async def progress_runner(completion_queue, matched_entries, unprobed_entries, unscanned_entries, final_results,
                          state_file_path):
    total_probes = len(unprobed_entries)
    total_scans = len(unscanned_entries)

    probes_finished = 0
    scans_finished = 0

    probe_stats = []

    time_slices = {}
    first_timestamp = -1
    last_timestamp = -1
    first_cpu_timestamp = -1
    last_cpu_timestamp = -1
    first_haccel_timestamp = -1
    last_haccel_timestamp = -1

    state_file_fp = None
    if state_file_path is not None:
        state_file_fp = open(state_file_path, 'a')

    while True:
        stage, result_entry = await completion_queue.get()
        if stage is None:
            completion_queue.task_done()
            if state_file_fp is not None:
                state_file_fp.close()
            break

        if stage == "probe":
            probes_finished += 1
            final_results[result_entry["filepath"]] = result_entry

            probe_duration = result_entry["probe_stats"]["end_clock"] - result_entry["probe_stats"]["start_clock"]
            current_probed = len([entry for entry in matched_entries if entry["probed"]])

            probe_stats.append(probe_duration)
            avg_probe_time = mean(probe_stats)

            estimated_total_remaining = 0.0
            if total_probes > probes_finished:
                estimated_total_remaining = avg_probe_time * (total_probes - probes_finished)
            formatted_time_remaining = str(timedelta(seconds=int(estimated_total_remaining)))

            logging.info(
                f"Probed {probes_finished} of {total_probes} ({current_probed / total_probes * 100:.0f}%) "
                f"in {probe_duration:.1f} seconds, "
                f"average {mean(probe_stats):.1f} seconds. "
                f"Estimated time remaining: {formatted_time_remaining}")

        elif stage.startswith("scan"):
            scans_finished += 1
            result_entry["complete"] = True

            runner_type = stage.split(':')[1]

            # Stats on this specific run
            scan_start = result_entry["scan_stats"]["start_clock"]
            scan_end = result_entry["scan_stats"]["end_clock"]
            scan_duration = scan_end - scan_start
            scan_size = result_entry["size"]
            scan_rate = scan_size / scan_duration

            # Calculate how many bytes we still have to go
            total_cpu_bytes_to_scan = sum(
                (entry["size"] for entry in unscanned_entries if not entry["haccel_compatible"]))
            total_haccel_bytes_to_scan = sum(
                (entry["size"] for entry in unscanned_entries if entry["haccel_compatible"]))

            # Calculate the first/last timestamps for our whole run
            if first_timestamp == -1 or scan_start < first_timestamp:
                first_timestamp = scan_start
            if last_timestamp == -1 or scan_end > last_timestamp:
                last_timestamp = scan_end
            if runner_type == "cpu" and first_cpu_timestamp == -1 or scan_start < first_cpu_timestamp:
                first_cpu_timestamp = scan_start
            if runner_type == "cpu" and last_cpu_timestamp == -1 or scan_end > last_cpu_timestamp:
                last_cpu_timestamp = scan_end
            if runner_type == "haccel" and first_haccel_timestamp == -1 or scan_start < first_haccel_timestamp:
                first_haccel_timestamp = scan_start
            if runner_type == "haccel" and last_haccel_timestamp == -1 or scan_end > last_haccel_timestamp:
                last_haccel_timestamp = scan_end

            # calculate time spent actually decoding.
            # this is not technically correct as it counts the overhead of queue tasks, which is admittedly negligible
            total_duration = last_timestamp - first_timestamp
            total_cpu_duration = last_cpu_timestamp - first_cpu_timestamp
            total_haccel_duration = last_haccel_timestamp - first_haccel_timestamp

            # Since we are doing concurrent things, we need to calculate rate a bit different
            # This will take the average bytes/sec of this run, then spread it out over a hashtable
            # that has keys of full seconds of clock time as keys. This way, we know the average bytes of data
            # that is being processed at every clock second. We can use this for actual rate calculations
            for second_slice in range(int(scan_start), int(scan_end)):
                slice_bytes_total, slice_bytes_cpu, slice_bytes_haccel = time_slices.setdefault(second_slice, (0, 0, 0))
                slice_bytes_total += scan_rate

                if runner_type == "cpu":
                    slice_bytes_cpu += scan_rate
                else:
                    slice_bytes_haccel += scan_rate

                time_slices[second_slice] = (slice_bytes_total, slice_bytes_cpu, slice_bytes_haccel)

            total_scan_rate = sum((t_bytes for t_bytes, c_bytes, h_bytes in time_slices.values())) / total_duration
            cpu_scan_rate = sum((c_bytes for t_bytes, c_bytes, h_bytes in time_slices.values())) / total_cpu_duration
            haccel_scan_rate = sum(
                (h_bytes for t_bytes, c_bytes, h_bytes in time_slices.values())) / total_haccel_duration

            estimated_total_remaining = 0
            if cpu_scan_rate > 0.0:
                estimated_total_remaining += (total_cpu_bytes_to_scan / cpu_scan_rate)
                if haccel_scan_rate > 0.0:
                    # haccel has no rate data, assume cpu speed
                    estimated_total_remaining += (total_haccel_bytes_to_scan / cpu_scan_rate)
            if haccel_scan_rate > 0.0:
                estimated_total_remaining += (total_haccel_bytes_to_scan / haccel_scan_rate)
                if cpu_scan_rate > 0.0:
                    # cpu has no rate data, assume haccel speed, not a great number, but better than nothing
                    estimated_total_remaining += (total_cpu_bytes_to_scan / haccel_scan_rate)

            logging.info(
                f"Scan completed ({runner_type}) in {scan_duration:.2f} seconds, "
                f"{human_readable_size(scan_rate)}/s "
                f"{os.path.basename(result_entry['filepath'])}")

            formatted_time_remaining = str(timedelta(seconds=int(estimated_total_remaining)))
            logging.info(f"Scanned {scans_finished}/{total_scans} ({scans_finished / total_scans * 100:.0f}%) "
                         f"| Rate: {human_readable_size(total_scan_rate)}/s "
                         f"| CPU {human_readable_size(cpu_scan_rate)}/s "
                         f"| HACCEL {human_readable_size(haccel_scan_rate)}/s"
                         f"| ETA: {formatted_time_remaining} seconds")

        if state_file_fp is not None:
            json.dump(result_entry, state_file_fp)
            state_file_fp.write('\n')

        completion_queue.task_done()


def create_new_result_entry(filepath):
    return {
        "filepath": filepath,
        "probed": False,
        "scanned": False,
        "complete": False,
        "size": os.path.getsize(filepath),
        "codec": None,
        "haccel_compatible": False,
        "scan_passed": False,
        "raw_scan_results": None,
        "scan_stats": None,
        "scan_has_warnings": None,
        "probe_passed": False,
        "probe_stats": None,
        "retry_scan_queue_size_bug": False,
        "force_cpu_only": False,
        "force_cpu_only_no_queue_size": False
    }


async def check_videos(search_paths, state_file_path=None, probe_concurrency=20, cpu_scan_concurrency=3,
                       haccel_scan_concurrency=5, is_dry_run=False):
    final_results = {}

    if state_file_path is not None and os.path.exists(state_file_path):
        logging.info(f'Reading in state file {state_file_path}')
        with open(state_file_path) as json_file:
            for i, line in enumerate(json_file):
                try:
                    entry = json.loads(line)
                    final_results[entry['filepath']] = entry
                except:
                    logging.error(f'could not load line {i} in state file, skipping')
            logging.debug(f'loaded {i} state changes from state file')
        logging.info(f'State file load complete, found {len(final_results)} records')

    logging.info(f'Searching for all video files in {":".join(search_paths)}')
    matched_files = [filepath for folderpath in search_paths for filepath in get_all_video_file_paths(folderpath)]
    for filepath in matched_files:
        if final_results.get(filepath, None) is None:
            final_results[filepath] = create_new_result_entry(filepath)

    matched_entries = [final_results[filepath] for filepath in matched_files]
    unprobed_entries = [entry for entry in matched_entries if not entry["probed"]]
    unscanned_entries = [entry for entry in matched_entries if not entry["scanned"]]

    logging.info(f'Found {len(matched_files)} matching video files, {len(unprobed_entries)} need probing')

    if not is_dry_run:
        completion_queue = asyncio.Queue()
        progress_consumer = asyncio.create_task(
            progress_runner(completion_queue, matched_entries, unprobed_entries, unscanned_entries, final_results,
                            state_file_path))

        # start our consumers for probe stage
        consumers = [asyncio.create_task(video_probe_consumer_runner(unprobed_entries, completion_queue))
                     for _ in range(probe_concurrency)]

        await asyncio.gather(*consumers)

        total_bytes = sum((entry["size"] for entry in unscanned_entries))
        logging.info(f'Starting full file scan of {len(unscanned_entries)} files, {human_readable_size(total_bytes)}')

        unscanned_entries.sort(key=lambda x: x["size"])

        consumers = [asyncio.create_task(video_scan_consumer_runner(unscanned_entries, False, completion_queue))
                     for _ in range(cpu_scan_concurrency)]

        consumers.extend([asyncio.create_task(video_scan_consumer_runner(unscanned_entries, True, completion_queue))
                          for _ in range(haccel_scan_concurrency)])

        await asyncio.gather(*consumers)

        completion_queue.put_nowait((None, None))

        await completion_queue.join()
        await asyncio.gather(progress_consumer)
    else:
        logging.info('Skipping probing/scanning, is a dry run')

    return final_results, matched_files


ffpemg_warning_regex = [re.compile(regex) for regex in ffmpeg_warning_patterns]
ffpemg_ignore_regex = [re.compile(regex) for regex in ffmpeg_ignore_patterns]


def classify_ffmpeg_message(message):
    for regex in ffpemg_ignore_regex:
        if regex.match(message):
            return "ignore"
    for regex in ffpemg_warning_regex:
        if regex.match(message):
            return "warning"
    # if a message is not a warning or ignore, treat as an error
    return "error"


def get_errors_warnings_from_ffmpeg_responses(raw_output):
    errors = []
    warnings = []
    for message in (raw_output or []):
        message_type = classify_ffmpeg_message(message)
        if message_type == "error":
            errors.append(message)
        elif message_type == "warning":
            warnings.append(message)

    return errors, warnings


def print_ffmpeg_messages(issues, truncate=True):
    if truncate is True or truncate == 1:
        sorted_issues = sorted(issue.strip() for issue in issues)
        grouped_issues = []
        cur_group = []
        for i, issue in enumerate(sorted_issues):
            if i == 0:
                cur_group = [issue]
                continue

            match_ratio = difflib.SequenceMatcher(None, cur_group[-1], issue).ratio()

            # if there is a match, add it to the group
            if match_ratio >= 0.9:
                cur_group.append(issue)
            else:
                grouped_issues.append(cur_group)
                cur_group = [issue]

        grouped_issues.append(cur_group)
        grouped_issues.sort(key=lambda x: len(x), reverse=True)

        for i, issue_group in enumerate(grouped_issues):
            if len(issue_group) > 1:
                print(f"{i}:\t({len(issue_group)}x) {issue_group[0]}")
            else:
                print(f"{i}:\t{issue_group[0]}")

    else:
        for i, issue in enumerate(issues):
            print(f"{i}:\t{issue}")


def print_results(final_results, matched_files, errors_level=False, warnings_level=False):
    result_count = len(matched_files)
    unscanned = 0
    unprobed = 0
    successful = 0
    warnings = 0
    failed = 0

    for filepath in matched_files:
        result = final_results[filepath]
        if not result["probed"]:
            unprobed += 1
        elif not result["scanned"]:
            unscanned += 1
        elif not result["probe_passed"]:
            failed += 1
            print(f"ffprobe failed: '{result['filepath']}'")
        elif not result["scan_passed"] or result["scan_has_warnings"]:
            ffmpeg_errors, ffmpeg_warnings = get_errors_warnings_from_ffmpeg_responses(result['raw_scan_results'])
            if len(ffmpeg_errors) > 0:
                failed += 1
                print(f"ffmpeg scan errors: '{result['filepath']}'")
                if errors_level is True or errors_level > 0:
                    print_ffmpeg_messages(ffmpeg_errors, errors_level)
            elif len(ffmpeg_warnings) > 0:
                warnings += 1
                print(f"ffmpeg scan warnings: '{result['filepath']}'")
                if warnings_level is True or warnings_level > 0:
                    print_ffmpeg_messages(ffmpeg_warnings, warnings_level)
        else:
            successful += 1

    print(
        f"Scanned {successful + failed} files "
        f"| Successfull: {successful} "
        f"| Warning: {warnings} "
        f"| Failed: {failed} "
        f"| Unprobed: {unprobed} "
        f"| Unscanned: {unscanned}"
        f"| Total: {result_count}")


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file-history', help='store state in this file to allow the script to be restarted')
    parser.add_argument('-d', '--dry-run', action='store_true',
                        help='don\'t do any probing or scanning, just dump results. Useful when working with state files')
    parser.add_argument('-p', '--probe-concurrency', type=int,
                        help='how many concurrent ffprobe jobs to run, these are very cheap', default=20)
    parser.add_argument('-c', '--cpu-scan-concurrency', type=int,
                        help='how many concurrent non-hardware acclerated ffmpeg scan jobs to run, these are very quite expensive',
                        default=1)
    parser.add_argument('-a', '--haccel-scan-concurrency', type=int,
                        help='how many concurrent hardware acclerated ffmpeg scan jobs to run, these are very quite expensive',
                        default=0)
    parser.add_argument('-e', '--errors', action='count',
                        help='Print out errors, use -ee to print out all errors and don\'t truncate repeated errors',
                        default=0)
    parser.add_argument('-w', '--warnings', action='count',
                        help='Print out warnings, use -ww to print out all warnings and don\'t truncate repeated errors',
                        default=0)
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Include debug logging, this can get very chatty')
    parser.add_argument('searchdirs', nargs='+')
    args = parser.parse_args()

    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, args.verbose + 1)]  # capped to number of levels

    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)s %(message)s")

    final_results, matched_files = asyncio.run(
        check_videos(args.searchdirs, args.file_history, args.probe_concurrency, args.cpu_scan_concurrency,
                     args.haccel_scan_concurrency, args.dry_run))

    print_results(final_results, matched_files, args.errors, args.warnings)

    end = time.time()
    rounded_end = "{0:.4f}".format(round(end - start, 4))
    logging.info("Script ran in about %s seconds" % (rounded_end))
