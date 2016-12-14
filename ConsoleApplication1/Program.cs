using System;
using System.Collections.Generic;




namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {

            //Тестовые данные
            double[][] XOR_Input = new double[4][]
            {
                new double[2]{0, 0},
                new double[2]{1, 0},
                new double[2]{0, 1},
                new double[2]{1, 1}
            };
            double[][] XOR_Output = new double[4][]
            {
                new double[1]{0},
                new double[1]{1},
                new double[1]{1},
                new double[1]{0}
            };

            //Создание нейронной сети с одним скрытым слоем: 4            
            NeuralNetworkAB<double> testNetwork = new NeuralNetworkAB<double>(new int[] { 2, 4, 1 });

            // SetValue - функция для преобразования входных значений
            testNetwork.SetValue = (x) => x;
            // GetValue - функция для преобразования выходных значений
            testNetwork.GetValue = (y) =>  y;      
                                       
            // Загружает все тестовые данные в сеть             
            testNetwork.Input = XOR_Input;
            testNetwork.Output = XOR_Output;

            // Данный код так же подгружает все тестовые данные в нейронную сеть
            /*
            for(int i = 0; i < XOR_Input.Length; i++)
            { 
                // Метод Add так же подгружает данные в сеть, но по одно одному массиву за раз
                testNetwork.Add(XOR_Input[i], XOR_Output[i]);
            }
            */

            // Обучаем сеть (если обучение прошло без ошибок, вернёт true)
            bool learned = testNetwork.Learn(1E-5);
            if (learned)
            {
                Console.WriteLine(string.Join(" , ", testNetwork.Run(XOR_Input[2])));
                bool saved = testNetwork.Save("test");
                if (saved)
                {
                    NeuralNetworkAB<double> B = NeuralNetworkAB<double>.LoadFromFile("test");
                    Console.WriteLine(string.Join(" , ", B.Run(XOR_Input[2])));
                }
                else
                {
                    Console.WriteLine("Don't saved, see debug");
                }
            }
            else
            {
                Console.WriteLine("Don't learned, see debug");
            }
            /*
            MNIST DB = new MNIST();
            DB.LoadDB(1000, 100);
            NeuralNetworkAB<byte> A = new NeuralNetworkAB<byte>(new int[] { 784, 793, 10 });//3920
            foreach (DigitImage i in DB.TrainingImages)
            {
                double[] output = new double[10] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                output[i.Label] = 1;
                A.Add(i.RawImage, output);
            }
            A.Learn();
            while (true)
            {
                try
                {
                    int r = Convert.ToInt32(Console.ReadLine());
                    Console.WriteLine(DB.TestImages[r].ToString());
                    double[] result = A.Run(DB.TestImages[r].RawImage);
                    List<double> res = new List<double>();
                    for (int i = 0; i < result.Length; i++)
                        if (result[i] == 1)
                            res.Add(i);
                    Console.WriteLine(string.Join(" , ", res.ToArray()));
                }
                catch
                {
                    break;
                }
            }
            A.Save("MNIST1000DB");
            */
            Console.ReadKey();
        }
    }
}
