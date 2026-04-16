Files for checkpoint 3

Audio sample.wav: 16kHz, mono audio.

Phase 0 tests:
-test1: far end audio played directly from laptop
-test2: far end audio played externally

Phase 1 tests:
-test1: simple nlms with the same parameters as stored in the algorithm
-test2: nlms with heuristic delay in sampling. Set by default to 60ms
-test3: nlms with automatic delay. Normally calculated as 200ms
-test4: simple nlms with no delay, also prints ERLE graph at the end
-test5: nlms with heuristic delay in sampling. Set by default to 60ms. Also prints graph at the end
-test6: nlms with automatic delay. Normally calculated as 200ms. Also prints graph at the end
-test7: 80 iterations over manually delayed nlms algorithm with varying parameters. More details in "Test graphs" folder
-test8: 80 iterations over automatically delayed nlms algorithm with varying parameters. More details in "Test graphs" folder
-test9: 80 iterations over manually delayed nlms algorithm (delay = 200ms) with varying parameters. More details in "Test graphs" folder. This algorithm in particular does NOT emit sound directly from the device speaker and relies on external audio.
-test10: Single iteration over manually delayed nlms algorithm (delay = 200ms) with preset parameters. This algorithm in particular does NOT emit sound directly from the device speaker and relies on external audio.
______________________________________________________________
Credits:
Clean sample from:  LibriSpeech ASR corpus, openslr.org

Noisy sample from:https://openslr.org/162/
  
@misc{mudasiru_2025, 
     title={{Multi-Speaker Separation and Background Speech Enhancement with Deep Learning}}, 
     author={Mudasiru, Rasheed}, 
     year={2025}, 
     howpublished={Undergraduate Project Dataset}, 
     institution={Federal University of Technology Minna}, 
     license={MIT}, 
     note={63-minute multi-speaker English conversation with background noise for speech processing research.} 
} 
