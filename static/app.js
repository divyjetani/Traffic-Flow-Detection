const form = document.querySelector("#upload-form");
const input = document.querySelector("#video-input");
const fileLabel = document.querySelector("#file-label");
const submitButton = document.querySelector("#submit-button");
const statusMessage = document.querySelector("#status");
const results = document.querySelector("#results");
const originalVideo = document.querySelector("#original-video");
const outputVideo = document.querySelector("#output-video");
const livePreview = document.querySelector("#live-preview");
const outputPlaceholder = document.querySelector("#output-placeholder");
const processingProgress = document.querySelector("#processing-progress");
const progressMeter = document.querySelector("#progress-meter");
const progressLabel = document.querySelector("#progress-label");
const downloadLink = document.querySelector("#download-link");
const outputName = document.querySelector("#output-name");
let previewUrl;
let activeJob = 0;
let previewFrame = -1;
let currentPreviewJob = "";
let usingLegacyPreview = false;

livePreview.addEventListener("load", () => {
  if (!outputVideo.hidden) return;
  livePreview.hidden = false;
  outputPlaceholder.hidden = true;
});

livePreview.addEventListener("error", () => {
  if (currentPreviewJob && !usingLegacyPreview) {
    usingLegacyPreview = true;
    livePreview.src = `/api/jobs/${encodeURIComponent(currentPreviewJob)}/preview`;
    return;
  }
  if (outputVideo.hidden && !outputPlaceholder.hidden) {
    outputPlaceholder.innerHTML = "<p>Could not load the annotated frame preview.</p>";
  }
});

outputVideo.addEventListener("canplay", () => {
  outputVideo.hidden = false;
  livePreview.hidden = true;
  outputPlaceholder.hidden = true;
});

outputVideo.addEventListener("error", () => {
  outputVideo.hidden = true;
  if (livePreview.src) {
    livePreview.hidden = false;
    statusMessage.textContent =
      "The annotated video is ready to download, but this browser could not play it. Showing its last annotated frame.";
    statusMessage.className = "status is-error";
  } else {
    outputPlaceholder.innerHTML = "<p>The video could not be played in this browser.</p>";
    outputPlaceholder.hidden = false;
  }
});

input.addEventListener("change", () => {
  const file = input.files[0];
  fileLabel.textContent = file ? file.name : "Choose a video";
  submitButton.disabled = !file;
  if (!file) return;

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  originalVideo.src = previewUrl;
  originalVideo.load();
  document.querySelector("#original-name").textContent = file.name;
  previewFrame = -1;
  currentPreviewJob = "";
  usingLegacyPreview = false;
  outputVideo.hidden = true;
  outputVideo.removeAttribute("src");
  livePreview.hidden = true;
  livePreview.removeAttribute("src");
  outputPlaceholder.hidden = false;
  outputPlaceholder.innerHTML =
    '<span class="placeholder-icon" aria-hidden="true">▶</span><p>Select a video and start processing to see annotated frames</p>';
  processingProgress.hidden = true;
  outputName.textContent = "";
  downloadLink.hidden = true;
  results.hidden = false;
  statusMessage.textContent = "Ready to process your video.";
  statusMessage.className = "status";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = input.files[0];
  if (!file) return;

  const jobSequence = ++activeJob;
  const body = new FormData(form);
  submitButton.disabled = true;
  input.disabled = true;
  statusMessage.textContent = "Uploading video…";
  statusMessage.className = "status is-working";
  outputPlaceholder.innerHTML = '<span class="spinner" aria-hidden="true"></span><p>Waiting for the first annotated frame…</p>';
  outputPlaceholder.hidden = false;
  processingProgress.hidden = false;
  progressMeter.removeAttribute("value");
  progressLabel.textContent = "Uploading video…";
  results.hidden = false;

  try {
    const response = await fetch("/api/jobs", { method: "POST", body });
    const created = await response.json();
    if (!response.ok) throw new Error(created.error || "Upload failed.");
    currentPreviewJob = created.job_id;
    usingLegacyPreview = false;
    statusMessage.textContent = "Video uploaded. Processing is starting…";
    await pollJob(created.job_id, jobSequence);
  } catch (error) {
    if (jobSequence !== activeJob) return;
    statusMessage.textContent = error.message;
    statusMessage.className = "status is-error";
    outputPlaceholder.innerHTML = "<p>Processing could not be started.</p>";
    livePreview.hidden = true;
    livePreview.removeAttribute("src");
    processingProgress.hidden = true;
    submitButton.disabled = false;
    input.disabled = false;
  }
});

async function pollJob(jobId, sequence) {
  while (sequence === activeJob) {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "Could not check processing status.");

    if (job.status === "queued" || job.status === "processing" || job.status === "encoding") {
      if (job.status === "queued") {
        statusMessage.textContent = "Your video is queued for processing…";
        progressLabel.textContent = "Waiting for processing…";
        progressMeter.removeAttribute("value");
      } else if (job.status === "encoding") {
        statusMessage.textContent = "Frames processed. Preparing video for browser playback…";
        progressMeter.value = 100;
        progressLabel.textContent = "Encoding browser-compatible video…";
      } else {
        statusMessage.textContent = "Processing video — annotated frames appear live…";
        if (job.frame > 0 && job.frame !== previewFrame) {
          previewFrame = job.frame;
          if (!usingLegacyPreview) {
            livePreview.src =
              `/api/jobs/${encodeURIComponent(jobId)}/preview.jpg?frame=${job.frame}`;
          }
        }
        if (job.progress === null) {
          progressMeter.removeAttribute("value");
          progressLabel.textContent = `Processed frame ${job.frame.toLocaleString()}`;
        } else {
          progressMeter.value = job.progress;
          progressLabel.textContent =
            `Frame ${job.frame.toLocaleString()} of ${job.total_frames.toLocaleString()} · ${job.progress}%`;
        }
      }
      await new Promise((resolve) => window.setTimeout(resolve, 600));
      continue;
    }

    if (job.status === "failed") throw new Error(`Processing failed: ${job.error}`);
    if (job.status !== "completed") throw new Error("The job returned an unknown status.");

    processingProgress.hidden = true;
    outputName.textContent = job.output_name;
    downloadLink.href = job.download_url;
    downloadLink.download = job.output_name;
    downloadLink.hidden = false;
    if (job.frame > 0 && job.frame !== previewFrame) {
      previewFrame = job.frame;
      if (!usingLegacyPreview) {
        livePreview.src =
          `/api/jobs/${encodeURIComponent(jobId)}/preview.jpg?frame=${job.frame}`;
      }
    }
    outputPlaceholder.innerHTML = '<span class="spinner" aria-hidden="true"></span><p>Loading annotated video…</p>';
    outputPlaceholder.hidden = false;
    outputVideo.src = job.video_url;
    outputVideo.load();
    statusMessage.textContent = "Processing complete. Loading the annotated video…";
    statusMessage.className = "status is-success";
    submitButton.disabled = false;
    input.disabled = false;
    return;
  }
}
