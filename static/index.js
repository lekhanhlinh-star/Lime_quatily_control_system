const webCam = document.getElementById('webCam')
var localstream;
const startWebCamBtn = document.getElementById('startWebCamBtn')
const stopWebCamBtn = document.getElementById('stopWebCamBtn')
let image_input = document.querySelector("#image_input");
let   photo =  document.querySelector("#photo");
const canvasElement = document.getElementById('canvasElement');
var uploaded_image = "";
var check_open_Cam=false

let sendFrameInterval = null
function startWebCam() {
    
        navigator.getUserMedia(
            { video: {} },
            stream => webCam.srcObject = stream,
            err => console.error(err)
          )
          check_open_Cam=true
          
}
function stopWebCam() {
    webCam.pause();
    webCam.srcObject  = null;
    clearInterval(sendFrameInterval);
    check_open_Cam=false
   
    
    
    
};


startWebCamBtn.addEventListener("click", startWebCam)
stopWebCamBtn.addEventListener("click", stopWebCam)
console.log("chay outside")

image_input.addEventListener("change", function(){
    console.log("ok")

    const reader = new FileReader(); 
    reader.addEventListener("load", ()=>{
        uploaded_image = reader.result;
        document.querySelector("#display_image").style.backgroundImage = `url(${uploaded_image})`;
        console.log("hahaha")
      
      
    });
    reader.readAsDataURL(this.files[0]);
    const file = this.files[0];
    const formData = new FormData();
    console.log("ok")
    formData.append('file', file);
    fetch('http://127.0.0.1:5000/upload-image', {
        method: 'POST',
        body: formData
      })
        .then(response => response.json())
        .then(data => {
            console.log(data.image_data)
          const img = new Image();
          img.src = data.image_data;
          
          photo.setAttribute("src",data.image_data);
          photo.onload = function() {
            const canvasContext = canvasElement.getContext('2d');
            // Draw the image on the canvas
            canvasContext.drawImage(data.image_data, 0, 0);
            console.log("photo load")
          };
          
        })
        .catch(error => console.error(error));
})





const webSocket = new WebSocket('ws://127.0.0.1:5000/predict');

// event listener for when the WebSocket connection is established
webSocket.addEventListener('open', function (event) {
    console.log('WebSocket connection is open');
});

// event listener for when the WebSocket connection is closed
webSocket.addEventListener('close', function (event) {
    console.log('WebSocket connection is closed');
});

// event listener for when a new frame is available from the webcam
webCam.addEventListener('play', function () {
    // set the canvas dimensions to match the video dimensions
    canvasElement.width = webCam.videoWidth;
    canvasElement.height = webCam.videoHeight;
    var localMedia = null;

    // send a new frame to the server every 100ms


    
    sendFrameInterval =setInterval(function () {
       
      
        // draw the current frame onto the canvas
        const canvasContext = canvasElement.getContext('2d');
        canvasContext.drawImage(webCam, 0, 0, canvasElement.width, canvasElement.height);

        // get the base64-encoded data of the canvas image
        const imageData = canvasElement.toDataURL('image/png');
         webSocket .onmessage = function(event) {
            photo.setAttribute("src",event.data)
                     
         
       };
        
        

        
        
        

        // send the image data to the server
    webSocket.send(imageData)
       
    }, 100);
});



