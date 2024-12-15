const baseUrl = window.location.origin;
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

// Start camera stream
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => console.error("Error accessing camera:", err));

// Handle real-time pose capture
document.getElementById('captureButton').addEventListener('click', async () => {
    const selectedPose = document.getElementById('poseSelect').value;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/png');

    try {
        const response = await fetch(`${baseUrl}/realtime`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData, pose: selectedPose })
        });
        const data = await response.json();

        if (data.error) {
            document.getElementById('realtimeResult').innerText = `Error: ${data.error}`;
        } else {
            document.getElementById('realtimeResult').innerHTML = `
                <h5>Prediction: ${data.prediction}</h5>
                <p>Feedback: ${data.feedback}</p>
            `;
        }
    } catch (error) {
        console.error("Error:", error);
    }
});
