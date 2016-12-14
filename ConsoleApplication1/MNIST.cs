using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
    public class DigitImage
    {
        private const int DIM_SIZE = 28;
        public static int SIZE = DIM_SIZE * DIM_SIZE;

        private byte[][] pixels;
        private byte label;

        public byte Label
        {
            get { return label; }
            set { label = value; }
        }

        public byte[][] Pixels
        {
            get { return pixels; }
            set { pixels = value; }
        }

        public double[] RawImage
        {
            get
            {
                double[] res = new double[SIZE];

                for (int i = 0; i < DIM_SIZE; i++)
                {
                    for (int j = 0; j < DIM_SIZE; j++)
                    {
                        if (Pixels[j][i] > 30)
                            res[i * DIM_SIZE + j] = 255;
                        else
                            res[i * DIM_SIZE + j] = 0;
                    }
                }

                return res;
            }
        }

        public DigitImage(byte[][] _pixels, byte _label)
        {
            pixels = new byte[28][];
            for (int i = 0; i < pixels.Length; i++)
                pixels[i] = new byte[28];

            for (int i = 0; i < 28; i++)
                for (int j = 0; j < 28; j++)
                    pixels[i][j] = _pixels[i][j];

            label = _label;
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    if (pixels[i][j] < 30)
                        s += " "; //white
                    else
                        s += "."; //black
                }
                s += "\n";
            }
            s += label.ToString();
            return s;
        }
    }
    public class ReadMNIST
    {
        private byte[][] pixles;
        private byte label;
        private string m_labelsPath;
        private string m_imagesPath;

        private int m_DBSize;

        public int DBSize
        {
            get { return m_DBSize; }
            set { m_DBSize = value; }
        }


        private List<DigitImage> m_Images = new List<DigitImage>();

        public List<DigitImage> Images
        {
            get { return m_Images; }
            set { m_Images = value; }
        }

        public ReadMNIST(string labelsPath, string imagesPath, int size)
        {
            m_labelsPath = labelsPath;
            m_imagesPath = imagesPath;
            DBSize = size;

            Update();
        }

        public void Update()
        {
            try
            {
                FileStream fsLabels = new FileStream(m_labelsPath, FileMode.Open);
                FileStream fsImages = new FileStream(m_imagesPath, FileMode.Open);
                BinaryReader brLabels = new BinaryReader(fsLabels);
                BinaryReader brImages = new BinaryReader(fsImages);

                //parse images
                int magic1 = brImages.ReadInt32();
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int nubCols = brImages.ReadInt32();

                //parse labels
                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

                Images.Clear();

                pixles = new byte[28][];
                for (int i = 0; i < pixles.Length; i++)
                    pixles[i] = new byte[28];

                //for imgaes
                for (int di = 0; di < DBSize; di++)
                {
                    for (int i = 0; i < 28; i++)
                    {
                        for (int j = 0; j < 28; j++)
                        {
                            byte b = brImages.ReadByte();

                            if (b > 30)
                            {
                                pixles[i][j] = 255; //(byte)brImages.ReadByte();
                            }
                            else
                            {
                                pixles[i][j] = 0;
                            }
                        }

                    }
                    label = brLabels.ReadByte();
                    DigitImage dImage = new DigitImage(pixles, label);
                    //Console.WriteLine(dImage.ToString());
                    //Console.ReadLine();

                    Images.Add(dImage);
                }

                fsImages.Close();
                fsLabels.Close();
                brImages.Close();
                brLabels.Close();

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }
    }
    class MNIST
    {
        private ReadMNIST _TrainingDB;
        private ReadMNIST _TestDB;

        public MNIST()
        {

        }

        public List<DigitImage> TrainingImages
        {
            get { return _TrainingDB.Images; }
        }
        public List<DigitImage> TestImages
        {
            get { return _TestDB.Images; }
        }

        public Boolean LoadDB(int trainSize, int testSize, string filesPath = "")
        {
            try
            {
                string testImagesPath = filesPath + "t10k-images.idx3-ubyte";
                string testLabelsPath = filesPath + "t10k-labels.idx1-ubyte";
                string trainingImagesPath = filesPath + "train-images.idx3-ubyte";
                string trainingLabelsPath = filesPath + "train-labels.idx1-ubyte";

                _TrainingDB = new ReadMNIST(trainingLabelsPath, trainingImagesPath, trainSize);
                _TestDB = new ReadMNIST(testLabelsPath, testImagesPath, testSize);
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return false;
            }
        }
        public Boolean LoadDB(string trainPath, string labelPath, int trainSize)
        {
            try
            {
                string trainingImagesPath = trainPath;
                string trainingLabelsPath = labelPath;

                _TrainingDB = new ReadMNIST(trainingLabelsPath, trainingImagesPath, trainSize);
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return false;
            }
        }
    }
}
