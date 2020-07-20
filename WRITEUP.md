# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

### Custom Layers implementation

When applying a custom layer for your pre-trained model in the Open Vino toolkit, add extensions to both of Model Optimizer, and Inference Engine.

#### Model optimizer

For each layer, the Model Optimizer initially derives data from the input model that includes model layer topology along with parameters, input and output format, etc. The model then is configured from the different known features of the layers, interconnects, and flow of data that comes partly from the layer process offering information like output shape for each layer. The optimized model is finally output to the model IR files required by the Inference Engine for running the model.

There are majorly two custom layer extensions required:

1. Custom Layer Extractor

Accountable for the identification of custom layer operation and the abstraction of the parameters for each custom layer instance. For instance the layer parameters are preserved and used by the layer operation before eventually emerging in the output IR. Usually the parameters of the input layer are unaffected.

2. Custom Layer Operation

Accountable for defining the attributes supported by the custom layer, and calculating the output shape from its parameters at each instance of the custom layer. 

#### Inference Engine

Each device plugin contains a library of optimized configurations to perform known layer operations that need to be expanded to perform a custom layer. The extension of custom layer is applied according to the targeted device:

1. Custom Layer CPU Extension

The CPU Plugin requires a compiled shared library (`.so` or `.dll` binary) to execute the custom layer on CPU.

2. Custom Layer GPU Extension

OpenCL source code (`.cl`) for the custom layer kernel that will be compiled to run on GPU together with a layer description file (`.xml`) provided for the custom layer kernel by GPU Plug-in.

### Reasons for handling custom layers

It becomes very important in industry challenges to also be able to transform custom layers because your teams may be designing something new or working on something and your system will need to know how to help custom layers to function smoothly.

One other popular use case would be to use the `lambda` layers.  Those layers are where you might add to your model architecture an arbitrary portion of code. You'd need support for these types of layers and custom layers are your way.

#### my sourse:

<details>
  <summary>Source</summary>
  https://docs.openvinotoolkit.org/
</details>

## Comparing Model Performance

There are two principal metrics to be used here.

1. Accuracy
2. Speed

Accuracy can be computed simply by analyzing the results with data labelled by humans. 

We can quantify the inference time for speed. 

After conversion, 

- Accuracy drops due to reduced floating point precision. 

- Freezing, fusion, and quantization increases speed. 

- Model size decreases, too.

|System |Accuracy (mAP)  |Time (ms)  |Size (MB)  |
|---|---|---|---|
|Pre-Conversion   |21   |55   |28   |
|Post-Conversion   |21   |60   |26   |

## Assess Model Use Cases

Some of the people counter app's potentially use cases are: 

1. Tracking of citizens movements during the COVID-19 shutdown in public areas like parks, banks, theme parks, cinemas and so on. 

2. Detecting intruders in enclosed spaces or private property. The device does so by triggering an alert when detecting a human within the field of view of the camera. 

3. This can be used on drones to be used by first responders to cover more ground in a short time , minimizing response time and in effect increasing the success rate of search and rescue operations carried out in wide areas.

4. Using cameras in a community coupled with the people counter software fitted with a model trained to identify the particular felon or a group of felons, scanning citywide or national areas for wanted felons.

## Assess Effects on End User Needs

One can use the People Counter software in many ways. You might for example place a camera in front of a store, bank, or train station. The software will quantify how many people there are in the facility. That figure can be contrasted with the number of people allowed inside the building. When too many people are there, then an alert will be sent. They could incorporate a pipeline of other models. Ex: Identification of a corona mask.

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
