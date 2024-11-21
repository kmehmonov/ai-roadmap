const canvas = document.getElementById('canvas');

ctx = canvas.getContext('2d');
document.width = window.innerWidth;
document.height = window.innerHeight;

// Variables to track drawing state
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Event listeners for mouse actions
canvas.addEventListener('mousedown', (e) => {
  isDrawing = true;
  [lastX, lastY] = [e.offsetX, e.offsetY];
});

canvas.addEventListener('mousemove', (e) => {
  if (!isDrawing) return; // Stop if not drawing
  ctx.beginPath();
  ctx.moveTo(lastX, lastY); // Start from the last position
  ctx.lineTo(e.offsetX, e.offsetY); // Draw to the current mouse position
  ctx.stroke(); // Render the line
  [lastX, lastY] = [e.offsetX, e.offsetY]; // Update the last position
});

canvas.addEventListener('mouseup', () => (isDrawing = false));
canvas.addEventListener('mouseout', () => (isDrawing = false));

// Configure pen properties
ctx.lineWidth = 8; // Pen width
ctx.lineCap = 'round'; // Smooth edges
ctx.strokeStyle = 'blue'; // Pen color

// Function to get base64 image data
function getImageData() {
  const imageData = canvas.toDataURL('image/png'); // Base64 format
  return imageData;
}

// Function to send image data to the server
function sendImageData() {
  const imageData = getImageData();
  fetch('http://localhost:5000/process-image', {
    // Change to your server endpoint
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image: imageData }),
  })
    .then((response) => response.json())
    .then((data) => {
        console.log('Response from server:', data)
        document.getElementById('digit').innerHTML = data.digit;
        document.getElementById('score').innerHTML = data.score;
    })
    .catch((error) => console.error('Error:', error));
}

function clearRect() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Example button to trigger the send
document
  .getElementById('sendDataButton')
  .addEventListener('click', sendImageData);

document.getElementById('clear').addEventListener('click', clearRect);
