using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;





namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            
            MNIST DB = new MNIST();
            DB.LoadDB("", 1000, 10000);
            NeuralNetworkAB<byte> A = new NeuralNetworkAB<byte>(new int[] { 784, 793, 10 });//3920
            foreach (DigitImage i in DB.TrainingImages)
            {
                short[] output = new short[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                output[i.Label] = 1;
                A.Add(i.RawImage, output);
            }
            A.Learn(0.001);
            Console.WriteLine(DB.TrainingImages[100].ToString());
            double[] result = A.Run(DB.TrainingImages[100].RawImage);
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
            Console.WriteLine(string.Join(" , ", result));
            Console.ReadKey();
        }
    }
}
