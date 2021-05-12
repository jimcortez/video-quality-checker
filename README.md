# Video Quality Check Script

This script solves a problem of scanning a large amount of video files for errors. Here are the basic steps

1. Find all video files in a given path
2. Run `ffprobe` to check for basic file structure and to identify video codec. This is pretty fast.
3. For each file that passed the probe, scan the file using ffmepg. This is similar to re-encoding the video, and is very slow
  * This can be done in parallel using the `-c/--cpu-scan-concurrency N` and `-a/--haccel-scan-concurrency N` flags
  * If hardware acceleration is enabled, the script will also use your gpu to offload this work.
4. Will report on the results from ffmpeg, classifying each message
  * A `warning` message is a ffmpeg complaint that does not affect the playback of a video
  * An `error` message is a ffmpeg error that possibly affects the playback.
  
 Notes:
 * This is a standalone script, there are no dependencies
 * I recommend you have ffmpeg 4.4+, this may require building it yourself
 * You should use a state file, this will allow your script to keep track of what is already done so you can start/stop at any time
 * Warnings are definitely warnings, but not all errors are fatal, there are too many to track
 
 ## Usage
 
 ### Basic
 
 This will run 1 cpu job at a time and print out results.
 
 ```bash
python video-checker.py -e -f video-checker-state.json ./video_files
```

* `-e` print out errors when script is complete, this will not print warnings
* `-f` create a file with 1 json payload per line. This tracks the progress of the script. When an existing
       state file is used, the script will not re-start any jobs it has already completed. You can use the same state
       file with multiple different runs with different parameters, reporting will be limited to the search directories
       given.``
* `./video_files` one or more directories (or specific file paths) to search

### With Concurrency

Same as above, except start scanning 3 files at a time. Keep an eye on your cpu usage and adjust to your liking

 ```bash
python video-checker.py -c 3 -f video-checker-state.json ./video_files
```
* `-c 3` number of concurrent cpu jobs to run. These are quite cpu-intensive jobs


### With Hardware Acceleration
Same as above, except also allow 2 concurrent jobs to use the GPU to do some work as well. 
This will have 3 CPU jobs and 2 GPU jobs.

 ```bash
python video-checker.py -c 3 -a 2 -f video-checker-state.json ./video_files
```
* `-a 2` number of concurrent hardware-accelerated workers to use. The optimal amount completely depends on your GPU
capabilities.

You should adjust the `haccel_codecs` code block at the top of the file to match your GPU's capabilities.
Current examples work with nvidia GPU's. Ffmpeg must be compiled with GPU support for this to work.

The script is smart enough for the CPU workers to scan non-hardware-acceleration compatible first. If there are no
more cpu-only jobs, those workers will help on the hardware-acceleration queue. The hardware acceleration workers will
quit if there are no hardware-acceleration-capable files to be scanned.

### Dry-runs (don't prob or scan)
If you want to look at the results so far that are in your state file, use the `-d` flag and it will skip any actual
scanning or probing.

 ```bash
python video-checker.py -d -w -e -f video-checker-state.json ./video_files
```
* `-d` dry run, don't do any probing or scanning, just skip to reporting
* `-w` print out warnings

### Getting raw output from ffmpeg
By default, the script will attempt to truncate similar error and warning messages (sometimes in the 1000's). To disable
this behavior:
 ```bash
python video-checker.py -d -ww -ee -f video-checker-state.json ./video_files
```
* `-ww` print out warnings, don't truncate
* `-ee` print out errors, don't truncate

### Helptext

```
$ python3 video-checker.py --help
usage: video-checker.py [-h] [-f FILE_HISTORY] [-d] [-p PROBE_CONCURRENCY] [-c CPU_SCAN_CONCURRENCY] [-a HACCEL_SCAN_CONCURRENCY] [-e] [-w] [-v]
                        searchdirs [searchdirs ...]

positional arguments:
  searchdirs

optional arguments:
  -h, --help            show this help message and exit
  -f FILE_HISTORY, --file-history FILE_HISTORY
                        store state in this file to allow the script to be restarted
  -d, --dry-run         don't do any probing or scanning, just dump results. Useful when working with state files
  -p PROBE_CONCURRENCY, --probe-concurrency PROBE_CONCURRENCY
                        how many concurrent ffprobe jobs to run, these are very cheap
  -c CPU_SCAN_CONCURRENCY, --cpu-scan-concurrency CPU_SCAN_CONCURRENCY
                        how many concurrent non-hardware acclerated ffmpeg scan jobs to run, these are very quite expensive
  -a HACCEL_SCAN_CONCURRENCY, --haccel-scan-concurrency HACCEL_SCAN_CONCURRENCY
                        how many concurrent hardware acclerated ffmpeg scan jobs to run, these are very quite expensive
  -e, --errors          Print out errors, use -ee to print out all errors and don't truncate repeated errors
  -w, --warnings        Print out warnings, use -ww to print out all warnings and don't truncate repeated errors
  -v, --verbose
```

## Troubleshooting
* check to be sure python can find the `ffprobe` and `ffmpeg` binaries. See the `ffmpeg_paths` configuration for the
paths that are 
