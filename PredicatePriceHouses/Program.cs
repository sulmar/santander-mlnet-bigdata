using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO.Compression;
using System.Linq;

namespace PredicatePriceHouses
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
            // 1. Context
            var context = new MLContext();

            var model = context.Model.Load("houses-model.zip", out _);

            var engine = context.Model.CreatePredictionEngine<HousingData, HousingPrediction>(model);

            HousingData housingData = new HousingData
            {
                City = "Warszawa",
                Floor = 2,
                Address = "ul. Wrocławska",
                Rooms = 3,
                Sq = 80,
                Year = 2020,
                Latitude = 52.224665768f,
                Longitude = 21.006499974f
            };

            var prediction = engine.Predict(housingData);

            Console.WriteLine($"{prediction.PredictedPrice:C2}");
        }

        private static void TrainModel()
        {
            const string filename = "./extract/Houses.csv";

            // Unzip
            if (!System.IO.File.Exists(filename))
                ZipFile.ExtractToDirectory("Houses.csv.zip", "./extract");


            // 1. Context
            var context = new MLContext();

            // 2. Load data
            IDataView dataView = context.Data.LoadFromTextFile<HousingData>(filename, hasHeader: true, separatorChar: ',');

            var preview = dataView.Preview();

            var split = context.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var trainData = split.TrainSet;
            var testData = split.TestSet;

            string[] featureColumns = trainData.Schema
                .Select(column => column.Name)
                .Where(columnName => columnName != "Label"
                    && columnName != nameof(HousingData.Lp).ToLower()
                    && columnName != nameof(HousingData.Id).ToLower()
                    && columnName != "city"
                    && columnName != "address"
                    )
                .ToArray();


            var featurizedCity = context.Transforms.Text.FeaturizeText(inputColumnName: "city", outputColumnName: "CityFeaturized");
            var featurizedAddress = context.Transforms.Text.FeaturizeText(inputColumnName: "address", outputColumnName: "AddressFeaturized");

            // 3. Transform data
            var pipeline = context.Transforms.DropColumns(nameof(HousingData.Lp).ToLower(), nameof(HousingData.Id).ToLower());

            var features = context.Transforms.Concatenate(outputColumnName: "Features", inputColumnNames: featureColumns);

            // 4. Choose algoritm
            var trainer = context.Regression.Trainers.LbfgsPoissonRegression();

            // 5. Train model

            var trainPipeline = pipeline
                .Append(featurizedCity)
                .Append(featurizedAddress)
                .Append(features)
                .Append(trainer);

            var trainedModel = trainPipeline.Fit(trainData);

            // 6. Evaluate model
            var predictions = trainedModel.Transform(testData);

            var metrics = context.Regression.Evaluate(predictions);

            Console.WriteLine($"R^2: {metrics.RSquared}");

            // 7. Deploy
            context.Model.Save(trainedModel, dataView.Schema, "houses-model.zip");
        }
    }

    // lp,address,city,floor,id,latitude,longitude,price,rooms,sq,year
    internal class HousingData
    {
        [ColumnName("lp"), LoadColumn(0)]
        public int Lp { get; set; }
        [ColumnName("address"), LoadColumn(1)]
        public string Address { get; set; }
        [ColumnName("city"), LoadColumn(2)] 
        public string City { get; set; }
        [ColumnName("floor"), LoadColumn(3)] 
        public float Floor { get; set; }
        [ColumnName("id"), LoadColumn(4)] 
        public float Id { get; set; }
        [ColumnName("latitude"), LoadColumn(5)]
        public float Latitude { get; set; }
        [ColumnName("longitude"), LoadColumn(6)] 
        public float Longitude { get; set; }
        [ColumnName("Label"), LoadColumn(7)] // <--- Label
        public float Price { get; set; }
        [ColumnName("rooms"), LoadColumn(8)] 
        public float Rooms { get; set; }
        [ColumnName("sq"), LoadColumn(9)] 
        public float Sq { get; set; }
        [ColumnName("year"), LoadColumn(10)] 
        public float Year { get; set; }

    }

    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
        
    }
}
