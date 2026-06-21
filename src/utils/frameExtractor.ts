/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Extracts key frames from an uploaded video file using an offscreen canvas.
 * @param videoUrl The object URL of the video file.
 * @param frameCount Number of frames to extract.
 * @returns Promise resolving to an array of base64 image strings.
 */
export function extractFramesFromVideo(videoUrl: string, frameCount: number = 5): Promise<{ frames: string[]; duration: number; thumbnail: string }> {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.src = videoUrl;
    video.crossOrigin = "anonymous";
    video.muted = true;
    video.playsInline = true;

    video.addEventListener("loadedmetadata", async () => {
      const duration = video.duration;
      const extractedBase64s: string[] = [];
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      // Set canvas size based on standard constraints or video aspect ratio
      canvas.width = 480;
      canvas.height = 270;

      // Extract frames at intervals
      const timePoints: number[] = [];
      const step = duration / (frameCount + 1);
      for (let i = 1; i <= frameCount; i++) {
        timePoints.push(Math.min(i * step, duration - 0.1));
      }

      try {
        for (const time of timePoints) {
          await seekToTime(video, time);
          if (ctx) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const base64 = canvas.toDataURL("image/jpeg", 0.85);
            extractedBase64s.push(base64);
          }
        }

        // Use the middle frame or first frame as a thumbnail
        const thumbnail = extractedBase64s[Math.floor(extractedBase64s.length / 2)] || extractedBase64s[0] || "";

        resolve({
          frames: extractedBase64s,
          duration,
          thumbnail,
        });
      } catch (err) {
        reject(err);
      } finally {
        video.src = "";
        video.load();
      }
    });

    video.addEventListener("error", (e) => {
      reject(new Error("Failed to load video file. Please ensure it is a valid MP4/WebM profile."));
    });
  });
}

function seekToTime(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    const onSeeked = () => {
      video.removeEventListener("seeked", onSeeked);
      resolve();
    };
    video.addEventListener("seeked", onSeeked);
    video.currentTime = time;
  });
}
