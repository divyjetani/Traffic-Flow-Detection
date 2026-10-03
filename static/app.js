const form = document.querySelector("#upload-form");
const input = document.querySelector("#video-input");
const fileLabel = document.querySelector("#file-label");
const submitButton = document.querySelector("#submit-button");
const statusMessage = document.querySelector("#status");
const results = document.querySelector("#results");
const originalVideo = document.querySelector("#original-video");
const outputVideo = document.querySelector("#output-video");
const outputPlaceholder = document.querySelector("#output-placeholder");
const downloadLink = document.querySelector("#download-link");
const outputName = document.querySelector("#output-name");
let previewUrl;
let activeJob = 0;

input.addEventListener("change", () => {
  const file = input.files[0];
  fileLabel.textContent = file ? file.name : "Choose a video";
  submitButton.disabled = !file;
  if (!file) return;

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  originalVideo.src = previewUrl;
  document.querySelector("#original-name").textContent = file.name;
  outputVideo.hidden = true;
  outputVideo.removeAttribute("src");
  outputPlaceholder.hidden = false;
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
  outputPlaceholder.innerHTML = '<span class="spinner" aria-hidden="true"></span><p>Uploading your video…</p>';
  outputPlaceholder.hidden = false;
  results.hidden = false;

  try {
    const response = await fetch("/api/jobs", { method: "POST", body });
    const created = await response.json();
    if (!response.ok) throw new Error(created.error || "Upload failed.");
    statusMessage.textContent = "Video uploaded. Processing is starting…";
    await pollJob(created.job_id, jobSequence);
  } catch (error) {
    if (jobSequence !== activeJob) return;
    statusMessage.textContent = error.message;
    statusMessage.className = "status is-error";
    outputPlaceholder.innerHTML = "<p>Processing could not be started.</p>";
    submitButton.disabled = false;
    input.disabled = false;
  }
});

async function pollJob(jobId, sequence) {
  while (sequence === activeJob) {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || "Could not check processing status.");

    if (job.status === "queued" || job.status === "processing") {
      statusMessage.textContent = job.status === "queued"
        ? "Your video is queued for processing…"
        : "Processing video — detecting and tracking vehicles…";
      await new Promise((resolve) => window.setTimeout(resolve, 1200));
      continue;
    }

    if (job.status === "failed") throw new Error(`Processing failed: ${job.error}`);
    if (job.status !== "completed") throw new Error("The job returned an unknown status.");

    outputVideo.src = job.video_url;
    outputVideo.hidden = false;
    outputPlaceholder.hidden = true;
    outputName.textContent = job.output_name;
    downloadLink.href = job.download_url;
    downloadLink.download = job.output_name;
    downloadLink.hidden = false;
    statusMessage.textContent = "Processing complete. Your annotated video is ready.";
    statusMessage.className = "status is-success";
    submitButton.disabled = false;
    input.disabled = false;
    return;
  }
}
