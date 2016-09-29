using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.IO;

using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Train;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Persist;

namespace NeuralNetwork
{
    /// <summary>
    /// Данный класс является абстрактной прослойкой между существующими библиотеками, созданный дабы упростить разработку нейронных сетей до максимума.
    /// </summary>
    /// <typeparam name="InputType">Любой числовой тип, с которым удобнее работать программисту</typeparam>
    public class NeuralNetworkAB<InputType>
    {
        BasicNetwork network = new BasicNetwork();

        private double[][] input = new double[0][];
        private double[][] output = new double[0][];
        private int outputCount = 1;


        /// <summary>
        /// Можно установить или получить значение input самому, а можно пополнить его через метод Add
        /// </summary>
        public InputType[][] Input
        {
            get
            {
                return ToInputTypeArray(input);
            }
            set
            {
                input = ToDoubleArray(value);
            }
        }

        /// <summary>
        /// Можно установить или получить значение output самому, а можно пополнить его через метод Add
        /// </summary>
        public InputType[][] Output
        {
            get
            {
                return ToInputTypeArray(output);
            }
            set
            {
                output = ToDoubleArray(value);
            }
        }

        /// <summary>
        /// Данный метод добавляет массив данных для обучения нейронной сети
        /// </summary>
        /// <param name="input">Входные данные для обучения</param>
        /// <param name="output">Идеальные значения, которые сеть должна давать на выходе</param>
        /// <param name="debug">Когда True, выводит значения в Debug</param>
        public void Add(double[] input, double[] output, bool debug = true)
        {
            try
            {
                if (output.Length != outputCount) throw new Exception("Количество выходных нейронов не равно количеству элементов во входном параметре");
                List<double[]> nowInMatrixInput = this.input.ToList();
                List<double[]> nowInMatrixOutput = this.output.ToList();

                nowInMatrixInput.Add(input);
                nowInMatrixOutput.Add(output);
                this.input = nowInMatrixInput.ToArray();
                this.output = nowInMatrixOutput.ToArray();
            }
            catch (OutOfMemoryException e)
            {
                Debug.WriteIf(debug, e);
                Environment.Exit(e.HResult);
            }
            catch (Exception e)
            {
                Debug.WriteLineIf(!debug, e);
                Console.WriteLine(e);
            }

        }

        /// <summary>
        /// Данный метод добавляет массив данных для обучения нейронной сети
        /// </summary>
        /// <typeparam name="I">Тип данных для обучения сети</typeparam>
        /// <typeparam name="O">Тип данных для обучения сети</typeparam>
        /// <param name="input">Входные данные для обучения</param>
        /// <param name="output">Идеальные значения, которые сеть должна давать на выходе</param>
        /// <param name="debug">Когда True, выводит значения в Debug</param>
        public void Add<I, O>(I[] input, O[] output, bool debug = true) where I : IComparable<I>
            where O : IComparable<O>
        {
            try
            {
                if (output.Length != outputCount) throw new Exception("Количество выходных нейронов не равно количеству элементов во входном параметре");
                List<double[]> nowInMatrixInput = this.input.ToList();
                List<double[]> nowInMatrixOutput = this.output.ToList();
                double[] inputDouble = Array.ConvertAll(input, (x) => (double)(dynamic)x);
                double[] outputDouble = Array.ConvertAll(output, (x) => (double)(dynamic)x);

                nowInMatrixInput.Add(inputDouble);
                nowInMatrixOutput.Add(outputDouble);
                this.input = nowInMatrixInput.ToArray();
                this.output = nowInMatrixOutput.ToArray();
            }
            catch (OutOfMemoryException e)
            {
                Debug.WriteIf(debug, e);
                Environment.Exit(e.HResult);
            }
            catch (Exception e)
            {
                Debug.WriteLineIf(!debug, e);
                Console.WriteLine(e);
            }

        }

        /// <summary>
        /// Метод для запуска нейронной сети
        /// </summary>
        /// <param name="input">Входные данные</param>
        /// <param name="round">Когда True, округляет значение</param>
        /// <param name="debug">Когда True, выводит значения в Debug</param>
        /// <returns>Возвращает массив double[], после обработки сети</returns>
        public double[] Run(double[] input, bool round = true, bool debug = true)
        {
            IMLData output;
            List<double> result = new List<double>();
            try
            {
                BasicMLData runUp = new BasicMLData(input);
                output = network.Compute(runUp);
                for (int i = 0; i < outputCount; i++)
                {
                    if (round)
                        result.Add(Math.Round(Convert.ToDouble(output[i])));
                    else
                        result.Add(Convert.ToDouble(output[i]));
                }
            }
            catch (Exception e)
            {
                Debug.WriteLineIf(debug, e);
                return null;
            }
            return result.ToArray();
        }

        /// <summary>
        /// Метод для запуска нейронной сети
        /// </summary>
        /// <param name="input">Входные данные</param>
        /// <param name="round">Когда True, округляет значение</param>
        /// <param name="debug">Когда True, выводит значения в Debug</param>
        /// <returns>Возвращает массив double[], после обработки сети</returns>
        public double[] Run(InputType[] input,bool round = true, bool debug = true)
        {
            IMLData output;
            List<double> result = new List<double>();
            try
            {
                BasicMLData runUp = new BasicMLData(Array.ConvertAll(input, (x) => (double)(dynamic)x));
                output = network.Compute(runUp);
                for (int i = 0; i < outputCount; i++)
                {
                    if(round)
                        result.Add(Math.Round(Convert.ToDouble(output[i])));
                    else
                        result.Add(Convert.ToDouble(output[i]));
                }
            }
            catch (Exception e)
            {
                Debug.WriteLineIf(debug, e);
                return null;
            }
            return result.ToArray();
        }

        /// <summary>
        /// Метод для обучения сети
        /// </summary>
        /// <param name="maxError">Максимальная погрешность результата</param>
        /// <param name="debug">Когда True, выводит значения в Debug</param>
        /// <returns>Если обучение пройдено успешно, возвращает True</returns>
        public bool Learn(double maxError = 0.01, bool debug = true)
        {
            try
            {
                IMLDataSet trainingSet = new BasicMLDataSet(input, output);
                IMLTrain train = new ResilientPropagation(network, trainingSet);
                do
                {
                    train.Iteration();
                    Debug.WriteLineIf(debug, train.Error);
                } while (train.Error >= maxError);
            }
            catch (OutOfMemoryException e)
            {
                Debug.WriteIf(debug, e);
                Environment.Exit(e.HResult);
            }
            catch (Exception e)
            {
                Debug.WriteLineIf(debug, e.Message);
                return false;
            }
            return true;
        }
        /// <summary>
        /// Метод для обучения сети
        /// </summary>
        /// <param name="iterations">Количество повторений при обучении сети</param>
        /// <param name="debug">Когда True, выводит значения в Debug</param>
        /// <returns>Если обучение пройдено успешно, возвращает True</returns>
        public bool Learn(int iterations, bool debug = true)
        {
            try
            {
                IMLDataSet trainingSet = new BasicMLDataSet(input, output);
                IMLTrain train = new ResilientPropagation(network, trainingSet);
                
                for (int i = 0; i < iterations; i++)
                {
                    train.Iteration();
                    Debug.WriteLineIf(debug, train.Error);
                }
            }
            catch (OutOfMemoryException e)
            {
                Debug.WriteIf(debug, e);
                Environment.Exit(e.HResult);
            }
            catch (Exception e)
            {
                Debug.WriteLine(e);
                return false;
            }
            return true;
        }

        /// <summary>
        /// Метод для обучения сети
        /// </summary>
        /// <param name="iterations">Количество повторений при обучении сети</param>
        /// <param name="maxError">Макслимальная погрешность сети</param>
        /// <param name="debug">Когда True, выводит значения в Debug</param>
        /// <returns>Если обучение пройдено успешно, возвращает True</returns>
        public bool Learn(int iterations, double maxError, bool debug = true)
        {
            try
            {
                IMLDataSet trainingSet = new BasicMLDataSet(input, output);
                IMLTrain train = new ResilientPropagation(network, trainingSet);

                for (int i = 0; i < iterations; i++)
                {
                    train.Iteration();
                    Debug.WriteLineIf(debug, train.Error);
                    if (train.Error <= maxError) break;
                }
            }
            catch (OutOfMemoryException e)
            {
                Debug.WriteIf(debug, e);
                Environment.Exit(e.HResult);
            }
            catch (Exception e)
            {
                Debug.WriteLine(e);
                return false;
            }
            return true;
        }

        /// <summary>
        /// Обучить сеть из файла
        /// </summary>
        /// <param name="filename">Имя файла</param>
        /// <param name="debug">Когда True, выводит значения в Debug</param>
        /// <returns>Если обучение пройдено успешно, возвращает True</returns>
        public bool LearnFromFile(string filename, bool debug)
        {
            try
            {
                network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(new FileInfo(filename));
            }
            catch (Exception e)
            {
                Debug.WriteLineIf(debug, e);
                return false;
            }
            Debug.WriteLineIf(debug, "Нейронная сеть успешно загружена.");
            return true;
        }

        /// <summary>
        /// Сохранить сеть в файл
        /// </summary>
        /// <param name="filename">Имя файла</param>
        /// <param name="debug">Когда True, выводит значения в Debug</param>
        /// <returns>Если сохранение пройдено успешно, возвращает True</returns>
        public bool Save(string filename, bool debug = true)
        {
            try
            {
                EncogDirectoryPersistence.SaveObject(new FileInfo(filename), network);
            }
            catch(FileLoadException e)
            {
                Debug.WriteLineIf(debug, "Ошибка сохранения файла: " + e);
                return false;
            }
            catch (Exception e)
            {
                Debug.WriteLineIf(debug, e);
                return false;
            }
            Debug.WriteLineIf(debug, "Нейронная сеть успешно сохранена.");
            return true;
        }


        /// <summary>
        /// Метод конвертирует InputType[][] в double[][]
        /// </summary>
        /// <param name="array">Двумерный массив</param>
        /// <returns>double[][]</returns>
        private double[][] ToDoubleArray(InputType[][] array)
        {
            List<double[]> result = new List<double[]>();
            List<double> arrForResult = new List<double>();
            foreach (InputType[] one in array)
            {
                double[] elem = Array.ConvertAll(one, (x) => (double)(dynamic)x);
                result.Add(elem);
            }
            return result.ToArray();
        }
        /// <summary>
        /// Метод конвертирует double[][] в InputType[][]
        /// </summary>
        /// <param name="array">Двумерный массив</param>
        /// <returns>InputType[][]</returns>
        private InputType[][] ToInputTypeArray(double[][] array)
        {
            List<InputType[]> result = new List<InputType[]>();
            List<InputType> arrForResult = new List<InputType>();
            foreach (double[] one in array)
            {
                InputType[] elem = Array.ConvertAll(one, (x) => (InputType)(dynamic)x);
                result.Add(elem);
            }
            return result.ToArray();
        }

        /// <summary>
        /// Инициализация сети 
        /// </summary>
        /// <param name="Neurals">Массив с нейронами</param>
        public NeuralNetworkAB(int[] Neurals)
        {
            
            outputCount = Neurals.Last();
            network.AddLayer(new BasicLayer(null, true, Neurals[0]));
            for (int i = 1; i < Neurals.Length - 1; i++)
            {
                network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, Neurals[i]));
            }
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, outputCount));
            network.Structure.FinalizeStructure();
            network.Reset();
        }
        /// <summary>
        /// Инициализация сети из файла
        /// </summary>
        /// <param name="filename">Имя файла</param>
        public NeuralNetworkAB(string filename,bool debug = true)
        {
            try
            {
                network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(new FileInfo(filename));
            }
            catch(Exception e)
            {
                Debug.WriteLineIf(debug, e);
            }
            Debug.WriteLineIf(debug, "Нейронная сеть успешно загружена.");
        }
    }
}
