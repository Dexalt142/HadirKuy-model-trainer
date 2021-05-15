const FaceAPI = require('face-api.js');
const Images = require('./images');
const fs = require('fs');
const Canvas = require('canvas');

let start = new Date().getTime();
console.log('Initializing.');

Promise.all([
    FaceAPI.nets.tinyFaceDetector.loadFromDisk(__dirname + '../../models'),
    FaceAPI.nets.faceLandmark68Net.loadFromDisk(__dirname + '../../models'),
    FaceAPI.nets.faceRecognitionNet.loadFromDisk(__dirname + '../../models')
]).then(async () => {
    let referenceData = [];
    let it = 1;
    FaceAPI.env.monkeyPatch({ Canvas: Canvas.Canvas, Image: Canvas.Image, ImageData: Canvas.ImageData });

    console.log('\nTraining model.');
    for (let i = 0; i < Images.length; i++) {
        let person = Images[i];
        for (let j = 0; j < person.image.length; j++) {
            const imageName = person.image[j];
            const loadedImage = await Canvas.loadImage(__dirname + `../../images/${imageName}`);
            const faceDescription = await FaceAPI.detectSingleFace(loadedImage, new FaceAPI.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptor();
            if (faceDescription) {
                const faceDescriptors = [faceDescription.descriptor];
                referenceData.push(new FaceAPI.LabeledFaceDescriptors(person.label, faceDescriptors));
                it++;
            }
        }
    }

    let stop = new Date().getTime();
    fs.writeFileSync('face_model.json', JSON.stringify(referenceData));
    console.log(`\nFinished processing ${it} image(s) in ${((stop - start) / 1000).toFixed()} second(s).`);
});