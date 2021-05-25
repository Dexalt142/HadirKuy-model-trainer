const FaceAPI = require('face-api.js');
const Person = require('./person');
const fs = require('fs');
const Canvas = require('canvas');

let start = new Date().getTime();
let success = 0;
let fail = 0;
let totalImages = 0;
console.log('Initializing.');

Promise.all([
    FaceAPI.nets.tinyFaceDetector.loadFromDisk(__dirname + '../../models'),
    FaceAPI.nets.faceLandmark68Net.loadFromDisk(__dirname + '../../models'),
    FaceAPI.nets.faceRecognitionNet.loadFromDisk(__dirname + '../../models')
]).then(async () => {
    let referenceData = [];
    FaceAPI.env.monkeyPatch({ Canvas: Canvas.Canvas, Image: Canvas.Image, ImageData: Canvas.ImageData });

    console.log('\nTraining model.');
    for (let i = 0; i < Person.length; i++) {
        const person = Person[i];
        const dirName = __dirname + '../../images/' + person.name.toLowerCase();
        const images = fs.readdirSync(dirName);
        for(let j = 0; j < images.length; j++) {
            const image = images[j];
            const fileExt = image.split('.')[1];
            if (fileExt === 'jpg') {
                console.log('[PROCESS] ' + image);
                const loadedImage = await Canvas.loadImage(dirName + '/' + image);
                const faceDescription = await FaceAPI.detectSingleFace(loadedImage, new FaceAPI.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptor();
                if (faceDescription) {
                    const faceDescriptors = [faceDescription.descriptor];
                    referenceData.push(new FaceAPI.LabeledFaceDescriptors(person.label, faceDescriptors));
                    success++;
                    console.log('[OK] ' + image);
                } else {
                    fail++;
                    console.log('[FAIL] ' + image);
                }
                totalImages++;
            }
        }
    }

    let stop = new Date().getTime();
    fs.writeFileSync('face_model.json', JSON.stringify(referenceData));
    console.log(`\nFinished processing ${totalImages} image(s) in ${((stop - start) / 1000).toFixed()} second(s).`);
    console.log(`Success: ${success}. Fail: ${fail}.`);
});