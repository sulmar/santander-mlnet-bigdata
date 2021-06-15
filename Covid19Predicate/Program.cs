using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System;

namespace Covid19Predicate
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            const string filename = "time_series_covid19_vaccine_global.csv";

            // dotnet add package Microsoft.ML

            // 1. Context
            var context = new MLContext();

            context.Log += (s, e) => Console.WriteLine(e.Message);

            // 2. Load data
            IDataView dataView = context.Data.LoadFromTextFile<CovidInput>(filename, separatorChar: ',', hasHeader: true);

            var preview = dataView.Preview();

            // 3. Transform

            var filteredDataView = context.Data.FilterByCustomPredicate<CovidInput>(dataView, input => input.Country != "Poland" && input.ProvinceState != "616");

            var filteredPreview = filteredDataView.Preview();


            // 4. Algorithm

            // dotnet add package Microsoft.ML.TimeSeries

            var pipeline = context.Forecasting.ForecastBySsa(
                outputColumnName: nameof(VaccinatedForecast.Forecast),
                inputColumnName: "People_fully_vaccinated",
                windowSize: 5,
                seriesLength: 10,
                trainSize: 100,
                horizon: 7); // 7 days

            // 5. Train model
            var trainedModel = pipeline.Fit(filteredDataView);

            // 6. Metrics

            // 7. Predict model

            // add using Microsoft.ML.Transforms.TimeSeries;

            var engine = trainedModel.CreateTimeSeriesEngine<CovidInput, VaccinatedForecast>(context);

            // Prediction/Forecasting for 7 days
            var forecasts = engine.Predict();

            foreach (var forecast in forecasts.Forecast)
            {
                Console.WriteLine(forecast);
            }
            
        }
    }

    // Country_Region,Date,Doses_admin,People_partially_vaccinated,People_fully_vaccinated,Report_Date_String,UID,Province_State
    // Afghanistan,2021-02-22,0,0,0,2021/02/22,4,

    internal class CovidInput
    {
        [LoadColumn(0), ColumnName("Country_Region")]
        public string Country { get; set; }

        [LoadColumn(1)]
        public DateTime Date { get; set; }

        [LoadColumn(2), ColumnName("Doses_admin")]
        public float DosesAdmin { get; set; }

        [LoadColumn(3), ColumnName("People_partially_vaccinated")]
        public float PeoplePartiallyVaccinated { get; set; }

        [LoadColumn(4), ColumnName("People_fully_vaccinated")]
        public float PeopleFullyVaccinated { get; set; }

        [LoadColumn(7), ColumnName("Province_State")]
        public string ProvinceState { get; set; }

    }

    public class VaccinatedForecast
    {
        public float[] Forecast { get; set; }
    }
}
