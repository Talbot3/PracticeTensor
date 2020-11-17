const tf = require('@tensorflow/tfjs-node');
tf.enableProdMode();
const Jimp = require('jimp');
const fetch = require('isomorphic-fetch');
const MOBILE_NET_URL = 'http://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/mobilenet/web_model/model.json';

function img2X(img) {
    return tf.tidy(()=>{
        return tf.browser.fromPixels(img)
                .toFloat()
                .sub(255 / 2)
                .div(255 / 2)
                .reshape([1, 224, 224, 3]);
    })
}

async function run() {
    // brower model
    const mobileNet = await tf.loadLayersModel(MOBILE_NET_URL);
    console.log(mobileNet.summary());
    const image = await Jimp.read('./hmbb.png');
    await image.resize(224, 224);
    
    const x = tf.complex([-2.25, 3.25], [4.75, 5.75]);
    tf.imag(x).print();
    // GraphModel
    // const graphModel =await tf.loadGraphModel("https://hub.tensorflow.google.cn/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1", { fromTFHub: true })
}
run();