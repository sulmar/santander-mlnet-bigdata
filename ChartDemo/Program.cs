using System;
using System.Collections.Generic;
using System.Linq;
using XPlot.Plotly;

namespace ChartDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            BarChartTest();

            ChartTest2();
        }

        private static void ChartTest2()
        {
            var data1 = new List<int> { 10, 54, 66, 77 };
            var data2 = new List<int> { 6, 4, 6, 9 };

            var data = data1.Zip(data2, (a, b) => Tuple.Create(a, b));

            Layout.Layout layout = new Layout.Layout { title = "Basic Bar Chart", direction = 45 };

            //PlotlyChart plotlyChart = new PlotlyChart();
            //plotlyChart.WithLayout(layout);

            // Chart.WithLayout(layout, ).Line(data).Show();

        }


        // dotnet add package XPlot.Plotly
        // dotnet add package FSharp.Core
        private static void BarChartTest()
        {
            var data = Enumerable.Range(1, 10).Select(v => v * 2);

            Chart.Bar(data).Show();

            Chart.Line(data).Show();
        }
    }
}
