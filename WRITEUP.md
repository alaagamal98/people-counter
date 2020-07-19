# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

### Custom Layers implementation

When implementing a custom layer for your pre-trained model in the
Open Vino toolkit, you will need to add extensions to both the Model 
Optimizer and the Inference Engine.

#### Model optimizer


The Model Optimizer first extracts information from the input model which includes the topology of the model layers along with parameters, input and output format, etc., for each layer. The model is then optimized from the various known characteristics of the layers, interconnects, and data flow which partly comes from the layer operation providing details including the shape of the output for each layer. Finally, the optimized model is output to the model IR files needed by the Inference Engine to run the model.

There are majorly two custom layer extensions required-

1. Custom Layer Extractor

Responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged, which is the case covered by this tutorial.

2. Custom Layer Operation

Responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters. The `--mo-op` command-line argument shown in the examples below generates a custom layer operation for the Model Optimizer.

#### Inference Engine


Each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. The custom layer extension is implemented according to the target device:

1. Custom Layer CPU Extension

A compiled shared library (`.so` or `.dll` binary) needed by the CPU Plugin for executing the custom layer on the CPU.

2. Custom Layer GPU Extension

OpenCL source code (`.cl`) for the custom layer kernel that will be compiled to execute on the GPU along with a layer description file (`.xml`) needed by the GPU Plugin for the custom layer kernel.

#### Using the model extension generator

The Model Extension Generator tool generates template source code files for each of the extensions needed by the Model Optimizer and the Inference Engine.

The script for this is available here-  `/opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py`

You could use this script in the following manner:

```
usage: You can use any combination of the following arguments:

Arguments to configure extension generation in the interactive mode:

optional arguments:
  -h, --help            show this help message and exit
  --mo-caffe-ext        generate a Model Optimizer Caffe* extractor
  --mo-mxnet-ext        generate a Model Optimizer MXNet* extractor
  --mo-tf-ext           generate a Model Optimizer TensorFlow* extractor
  --mo-op               generate a Model Optimizer operation
  --ie-cpu-ext          generate an Inference Engine CPU extension
  --ie-gpu-ext          generate an Inference Engine GPU extension
  --output_dir OUTPUT_DIR
                        set an output directory. If not specified, the current
                        directory is used by default.
```

### Reasons for handling custom layers

In industry problems it becomes very important to be able to convert custom layers as your teams might be developing something new or researching on something and your application to work smoothly you would need to know how you could have support for custom layers.

Another common use case would be when using `lambda` layers. These layers are where you could add an arbitary peice of code to your model implementation. You would need to have support for these kind of layers and custom layers is your way 

#### my sourse:
<details>
  <summary>Source</summary>
  https://docs.openvinotoolkit.org/
</details>

## Comparing Model Performance

There are two main metrics that are to be seen here.
1. Accuracy
2. Speed

Accuracy can be simply measured by comparing the results with human labelled data.

For speed we can measure the inference time.

After conversion, 

- Accuracy decreases because of decrease in floating point precision.

- Speed increases because of freezing, fusion and quantization.

- The size of the model also decreases.

|System |Accuracy (mAP)  |Time (ms)  |Size (MB)  |
|---|---|---|---|
|Pre-Conversion   |21   |55   |28   |
|Post-Conversion   |21   |60   |26   |


## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

1. Monitoring of civilian movements within public places such as parks, banks, theme parks, cinemas and so on during the period of the COVID-19 lockdown.

2. To detect intruders in restricted areas or private properties. The app does this by raising an alarm when it detects a person within the camera's range of vision.

3. It can be used on drones to be deployed by first responders to cover a lot of ground in a short period of time, reducing response time and in turn increase the rate of success of search and rescue missions conducted in large areas.

4. Searching city-wide or nation-wide areas for wanted felons using cameras in a city combined with the people counter app equipped with a model trained to detect that individual felon or a group of felons.


## Assess Effects on End User Needs

The People Counter app can be used in many ways. For example, you could put a camera in front of a shop, bank, or train station. The app can count how many people are in the facility. This amount can be compared to the number of people allowed in the facility. If there are too many people, an alarm is sent. A pipeline of other models could be implemented. Ex: Corona mask recognition.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
