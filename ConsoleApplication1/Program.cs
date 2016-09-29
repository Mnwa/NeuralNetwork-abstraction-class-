using System;
using System.Collections.Generic;




namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            
            MNIST DB = new MNIST();
            DB.LoadDB("", 60000, 10000);
            NeuralNetworkAB<byte> A = new NeuralNetworkAB<byte>(new int[] { 784, 793, 10 });//3920
            foreach (DigitImage i in DB.TrainingImages)
            {
                double[] output = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                output[i.Label] = 1;
                A.Add(i.RawImage, output);
            }
            A.Learn(1E-4);
            A.Save("mnistDB.data");
            Console.WriteLine(DB.TrainingImages[300].ToString());
            double[] result = A.Run(DB.TrainingImages[300].RawImage);
            List<double> res = new List<double>();
            for(int i = 0; i < result.Length; i++)
            {
                if (result[i] == 1)
                    res.Add(i);
            }
            /*
            double[][] XOR_I = new double[4][] {
                new double[2] { 255, 255},
                new double[2] { 200, 200 },
                new double[2] { 155, 155 },
                new double[2] { 100, 100 }
            };
            double[][] XOR_O = new double[4][] {
                new double[1] { 1 },
                new double[1] { 0.66 },
                new double[1] { 0.33 },
                new double[1] { 0 }
            };
            NeuralNetworkAB<double> A = new NeuralNetworkAB<double>(new int[] { 2, 3, 1 });
            A.Input = XOR_I;
            A.Output = XOR_O;
            A.Learn(0.01);
            
            double[] result = A.Run(new double[2] { 200, 200 });*/
            Console.WriteLine(string.Join(" , ", res.ToArray()));
            Console.ReadKey();
        }
    }
}
