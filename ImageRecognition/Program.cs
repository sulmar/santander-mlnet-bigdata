using Microsoft.ML;
using Microsoft.ML.Transforms.Image;
using System;
using System.Drawing;
using Microsoft.ML.TensorFlow;
using System.Collections.Generic;
using Microsoft.ML.Data;
using System.IO;
using System.Linq;

namespace ImageRecognition
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            // TrainModel();

            UseTrainedModel();

        }

        private static void UseTrainedModel()
        {
            

            string filename = @"C:\Users\marci\OneDrive\Obrazy\apple-1239300_1280.jpg";

            // Context
            var context = new MLContext();

            // Load Model
            var model = context.Model.Load("image-model.zip", out _);

            // Engine
            var engine = context.Model.CreatePredictionEngine<ImageInputData, ImageLabelPredictions>(model);

            // Prediction
            var bitmap = (Bitmap) Bitmap.FromFile(filename);

            ImageInputData input = new ImageInputData { Image = bitmap };

            var prediction = engine.Predict(input);

            var labels = File.ReadAllLines(@"TFInceptionModel\imagenet_comp_graph_label_strings.txt");


            var predictionLabels = prediction.PredictionLabels.OrderByDescending(p => p).Take(5);

            foreach (var predictionLabel in predictionLabels)
            {
                var labelIndex = prediction.PredictionLabels.AsSpan().IndexOf(predictionLabel);

                var classifiedLabel = labels[labelIndex];

                Console.WriteLine($"Image from {filename} predicted as {classifiedLabel} with probability {predictionLabel:P2}");
            }

        }

        private static void TrainModel()
        {
            const string tensorFlowModelFilePath = @"TFInceptionModel\tensorflow_inception_graph.pb";

            // dotnet add package Microsoft.ML.ImageAnalytics

            // 1. Context
            var context = new MLContext();

            // Transformacja
            var pipeline = context.Transforms.ResizeImages(
                "input",
                imageWidth: 224,
                imageHeight: 224,
                nameof(ImageInputData.Image))
                .Append(context.Transforms.ExtractPixels(
                    "input",
                    interleavePixelColors: true,
                    offsetImage: 117
                    ))
                .Append(context.Model
                    .LoadTensorFlowModel(tensorFlowModelFilePath)  // dotnet add package Microsoft.ML.TensorFlow
                    .ScoreTensorFlowModel(
                        outputColumnNames: new[] { "softmax2" },
                        inputColumnNames: new[] { "input" },
                        addBatchDimensionInput: true));

            // Uwaga: dotnet add package SciSharp.TensorFlow.Redist

            // Train model
            IDataView emptyTrainingSet = context.Data.LoadFromEnumerable(new List<ImageInputData>());

            ITransformer model = pipeline.Fit(emptyTrainingSet);

            context.Model.Save(model, null, "image-model.zip");
        }
    }

    public class ImageInputData
    {
        [ImageType(224, 244)]
        public Bitmap Image { get; set; }
    }

    public class ImageLabelPredictions
    {
        [ColumnName("softmax2")]
        public float[] PredictionLabels { get; set; }
    }
}
