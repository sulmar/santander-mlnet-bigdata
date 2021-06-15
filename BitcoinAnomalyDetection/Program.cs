using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BitcoinAnomalyDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            // https://www.coindesk.com/price/bitcoin


           // AnomalyDetectionTest();

            PredictPriceTest();

        }

        private static void PredictPriceTest()
        {
            // 1. Context

            var context = new MLContext();

            // 2. Load data
            IDataView dataView = context.Data.LoadFromTextFile<CurrencyModel>("bitcoinrates.csv", hasHeader: true, separatorChar: ',');

            var pipeline = context.Forecasting.ForecastBySsa(
                nameof(PricePrediction.Predictions),
                nameof(CurrencyModel.Close),
                windowSize: 5,
                seriesLength: 10,
                trainSize: 100,
                horizon: 2);

            var trainedModel = pipeline.Fit(dataView);

            var engine = trainedModel.CreateTimeSeriesEngine<CurrencyModel, PricePrediction>(context);

            var forecasts = engine.Predict(3);

            foreach (var forecast in forecasts.Predictions)
            {
                Console.WriteLine(forecast);
            }

        }

        private static void AnomalyDetectionTest()
        {
            // 1. Context

            var context = new MLContext();

            // 2. Load data
            IDataView dataView = context.Data.LoadFromTextFile<CurrencyModel>("bitcoinrates.csv", hasHeader: true, separatorChar: ',');

            var preview = dataView.Preview();

            // dotnet add package Microsoft.ML.TimeSeries

            var pipeline = context.Transforms.DetectSpikeBySsa(
                outputColumnName: nameof(SpikeAnomaly.Anomalies),
                inputColumnName: nameof(CurrencyModel.Close),
                confidence: 98.0,
                trainingWindowSize: 90,
                seasonalityWindowSize: 30,
                pvalueHistoryLength: 20);

            var transformatedData = pipeline.Fit(dataView).Transform(dataView);


            var anomalies = context.Data.CreateEnumerable<SpikeAnomaly>(transformatedData, reuseRowObject: false).ToList();

            var prices = dataView.GetColumn<float>(nameof(CurrencyModel.Close)).ToArray();
            var dates = dataView.GetColumn<DateTime>(nameof(CurrencyModel.Date)).ToArray();


            for (int i = 0; i < anomalies.Count; i++)
            {
                if (anomalies[i].Anomalies[0] == 1) // Anomalia
                {
                    Console.WriteLine($"{dates[i]}\t{prices[i]}");
                }
            }
        }
    }

    // Date,Open,High,Low,Close,Volume,Market Cap
    internal class CurrencyModel
    {
        [LoadColumn(0)]
        public DateTime Date { get; set; }

        [LoadColumn(1)]
        public float Open { get; set; }

        [LoadColumn(2)]
        public float High { get; set; }
        [LoadColumn(3)]
        public float Low { get; set; }
        [LoadColumn(4)]
        public float Close { get; set; }
        [LoadColumn(5)]
        public float Volume { get; set; }
        [LoadColumn(6)]
        public float MarketCap { get; set; }

    }

    public class SpikeAnomaly
    {
        [VectorType(2)]
        public double[] Anomalies { get; set; }
    }

    public class PricePrediction
    {
        [VectorType(2)]
        public float[] Predictions { get; set; }
    }
}
