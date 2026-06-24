let running=false
let countdown=4
let currentPhase="Idle"

function start(){

fetch('/start')
running=true

startMic()
checkStatus()

}

function stop(){

fetch('/stop')
running=false

}


function checkStatus(){

if(!running) return

fetch('/status')
.then(res=>res.json())
.then(data=>{

// Status
document.getElementById("status").innerText =
data.emergency ? "🚨 Emergency Detected" : "Monitoring"


// Reset timer when phase changes
if(currentPhase !== data.phase){
currentPhase = data.phase
countdown = data.timer
startCountdown()
}


// Logs
document.getElementById("logs").innerText =
data.logs.join("\n")


// Emergency Alert
if(data.emergency){

document.getElementById("emergency").innerText =
"🚨 ALERT SENT"

}else{

document.getElementById("emergency").innerText = ""

}

})

setTimeout(checkStatus,1000)

}


function startCountdown(){

if(!running) return

document.getElementById("phase").innerText =
"Phase: " + currentPhase + " (" + countdown + ")"

countdown--

if(countdown >= 0){
setTimeout(startCountdown,1000)
}

}


//////////////////////////////////////////////////
// MICROPHONE WAVEFORM
//////////////////////////////////////////////////

const canvas=document.getElementById("wave")
const ctx=canvas.getContext("2d")

canvas.width=900
canvas.height=150

let audioContext
let analyser
let dataArray

function startMic(){

navigator.mediaDevices.getUserMedia({audio:true})
.then(stream=>{

audioContext=new AudioContext()

const source=
audioContext.createMediaStreamSource(stream)

analyser=audioContext.createAnalyser()

source.connect(analyser)

analyser.fftSize=2048

const bufferLength=analyser.frequencyBinCount

dataArray=new Uint8Array(bufferLength)

draw()

})

}

function draw(){

requestAnimationFrame(draw)

if(!analyser) return

analyser.getByteTimeDomainData(dataArray)

ctx.fillStyle="#000"
ctx.fillRect(0,0,canvas.width,canvas.height)

ctx.lineWidth=2
ctx.strokeStyle="#00ffd5"

ctx.beginPath()

let sliceWidth=canvas.width/dataArray.length
let x=0

for(let i=0;i<dataArray.length;i++){

let v=dataArray[i]/128.0
let y=v*canvas.height/2

if(i===0){

ctx.moveTo(x,y)

}else{

ctx.lineTo(x,y)

}

x+=sliceWidth

}

ctx.lineTo(canvas.width,canvas.height/2)

ctx.stroke()

}