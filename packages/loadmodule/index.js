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

async function run2() {
    const model = tf.sequential();
    model.add(
        tf.layers.dense({ units: 100, activation: 'relu', inputShape: [200] }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd',
        metrics: ['MAE']
    });

    // Generate some random fake data for demo purpose.
    const xs = tf.randomUniform([10000, 200]);
    const ys = tf.randomUniform([10000, 1]);
    const valXs = tf.randomUniform([1000, 200]);
    const valYs = tf.randomUniform([1000, 1]);

    // Start model training process.
    await model.fit(xs, ys, {
        epochs: 100,
        validationData: [valXs, valYs],
        // Add the tensorBoard callback here.
        callbacks: tf.node.tensorBoard('/tmp/fit_logs_1')
    });
}

run2();