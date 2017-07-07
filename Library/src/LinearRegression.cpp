#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
using namespace std;
//Constants
const double doubleMin = numeric_limits<double>::lowest();
const double doubleMax = numeric_limits<double>::max();

//Global Variables
int numFeature;
int order;
int numTrainingData;
int numCol;
int numRow;
double stepSize = 0.05;
int maxIter = 20000;
string fname;
vector<vector<int> > featureSeries;
vector<vector<double> > X;
vector<vector<double> > input;
vector<vector<double> > featureScales; //featureScales[k][0] = avgFeature, featureScales[k][1] = range
vector<double> y;
vector<double> theta;

void featureSeriesGeneration(vector<int> cur);
void featureInput(ifstream *inputFile);
void featureVectorGeneration();
void featureScaling();
double computeCost();
double computeGrad(int featureIndex);
double compute_y(vector<double> x);
void gradientDescent();

int main(int argc, char** argv)
{
  numTrainingData = atoi(argv[1]);
  numFeature = atoi(argv[2]);
  order = atoi(argv[3]);
  fname = argv[4];
  vector<int> cur;
  y.resize(numTrainingData);
  featureSeriesGeneration(cur);
  for(int i=0; i<featureSeries.size(); ++i)
  {
    for(int j=0; j<featureSeries[i].size(); ++j)
    {
      cout<<featureSeries[i][j];
    }
    cout<<endl;
  }
  ifstream inputFile;
  inputFile.open(fname);
  featureInput(&inputFile);
  featureVectorGeneration();
  for(int i=0; i<X.size(); ++i)
  {
    for(int j=0; j<featureSeries.size(); ++j)
    {
      cout<<X[i][j]<<" ";
    }
    cout<<endl;
  }
  featureScaling();
  for(int i=0; i<X.size(); ++i)
  {
    for(int j=0; j<featureSeries.size(); ++j)
    {
      cout<<X[i][j]<<" ";
    }
    cout<<endl;
  }
  theta.resize(featureSeries.size(), 0.0);
  gradientDescent();
  cout<<computeCost()<<endl;
  vector<double> test(2);
  test[0] = 1; test[1] = (10-featureScales[1][0])/featureScales[1][1];
  cout<<compute_y(test)<<endl;
  return 0;
}

void featureSeriesGeneration(vector<int> cur) //Generation powers for creating feature cross terms
{
  if(cur.size()==numFeature)
  {
    featureSeries.push_back(cur);
    return;
  }
  int sumOrder = accumulate(cur.begin(), cur.end(), 0);
  for(int i=0;i<=order-sumOrder;++i)
  {
    vector<int> curCopy(cur);
    curCopy.push_back(i);
    featureSeriesGeneration(curCopy);
  }
}

void featureInput(ifstream *inputFile) //Data input
{
  for(int i=0; i<numTrainingData; ++i)
  {
    vector<double> cur(numFeature);
    for(int j=0; j<numFeature; ++j)
    {
      *inputFile>>cur[j];
    }
    input.push_back(cur);
    *inputFile>>y[i];
  }
}

void featureVectorGeneration() //Generating feature cross terms
{
  for(int k=0; k<numTrainingData; k++)
  {
    vector<double> cur(featureSeries.size());
    for(int i=0; i<featureSeries.size(); ++i)
    {
      double value = 1.0;
      for(int j=0; j<featureSeries[i].size(); ++j)
      {
        value *= pow(input[k][j], featureSeries[i][j]);
      }
      cur[i] = value;
    }
    X.push_back(cur);
  }
}

void featureScaling() //Scale to -0.5 ~ 0.5
{
  vector<double> scale(2);
  scale[0] = 1; scale[1] = 0; featureScales.push_back(scale);
  for(int i=1; i<featureSeries.size(); ++i)
  {
    vector<double> scale(2);
    double min = doubleMax, max = doubleMin;
    double sum = 0.0;
    for(int j=0; j<numTrainingData; ++j)
    {
      if(X[j][i]<min) min = X[j][i];
      if(X[j][i]>max) max = X[j][i];
      sum+=X[j][i];
    }
    scale[0] = sum/numTrainingData;
    scale[1] = max - min;
    featureScales.push_back(scale);
    for(int j=0; j<numTrainingData; ++j)
    {
      X[j][i]-=scale[0];
      X[j][i]/=scale[1];
    }
  }
}

double compute_y(vector<double> x)
{
  double _y = 0.0;
  for(int i=0; i<featureSeries.size(); ++i)
  {
    _y+=theta[i]*x[i];
  }
  return _y;
}

double computeCost() //Mean squared cost
{
  double cost = 0.0;
  for(int i=0; i<numTrainingData; ++i)
  {
    double _y = compute_y(X[i]);
    cost+=(_y-y[i])*(_y-y[i]);
  }
  return cost/2/numTrainingData;
}

double computeGrad(int featureIndex)
{
  double grad = 0.0;
  for(int i=0; i<numTrainingData; ++i)
  {
    double _y = compute_y(X[i]);
    grad += (_y-y[i])*X[i][featureIndex];
  }
  return grad/numTrainingData;
}

void gradientDescent()
{
  for(int j=0; j<maxIter; ++j)
  {
    for(int i=0; i<theta.size(); ++i)
    {
      theta[i]-=stepSize*computeGrad(i);
    }
  }
}
