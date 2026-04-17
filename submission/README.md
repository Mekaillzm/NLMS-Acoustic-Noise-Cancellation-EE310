NLMS acoustic echo cancellation EE310

Group 15 - EE310

Contains:
main_algorithm.py - the algorithm for NLMS that is imported to run the code
realtime_demo.m - Checkpoint 3 code for realtime demonstration. Please run this.

checkpoint_presentations/videos/demo_video
The demonstration video for the code.

Phase information for realtime_demo:

Phase 0, Baseline: Run the system with adaptation disabled (weights fixed at zero). We should hear the raw echo in the room. This confirms that echo is present and that your system is not relying on OS-level cancellation.

Phase 1, Convergence: Enable adaptation and remain silent. The echo should fade noticeably within a few seconds as the filter converges. Be prepared to explain what is happening to the weights during this phase and why convergence speed depends on \mu and the far-end signal characteristics.

Phase 2, Double-Talk: While the far-end is playing and the filter is converged, speak naturally into the microphone. Your voice should be audible at the output while the echo remains suppressed. If the echo returns during your speech, your Geigel detector is not working correctly. Speak at a natural volume.

Phase 3, Path Change: While the system is converged, physically change the acoustic environment, that is, place a book in front of the speaker, rotate the laptop, or cover the microphone partially and then uncover it. You will hear a brief burst of residual echo as the old filter weights become invalid, followed by re-convergence to the new path. Be ready to explain why this happens and what property of NLMS allows recovery.
