#include "nGram.hpp"

int main(void)
{
  NgramModel ngModel;
  string s;

  //cout << "sizeof(string) c++ string=" << sizeof(string) << endl;

  if((sizeof(U16) * 8) != 16){
    cout << "ERROR sizeof U16 = " << (sizeof(U16) * 8) << " not equal to 16 bits on this system" << endl;
  }
  else if((sizeof(U32) * 8) != 32){
    cout << "ERROR sizeof U32 = " << (sizeof(U32) * 8) << " not equal to 32 bits on this system" << endl;
  }
  else if((sizeof(U64) * 8) != 64){
    cout << "ERROR sizeof U64 = " << (sizeof(U64) * 8) << " not equal to 64 bits on this system" << endl;
  }
  else{
    string training = "../../oanc_SlateTrainData.txt";
    ngModel.Train(training);
    string testing = "../../oanc_SlateTestData.txt";
    ngModel.Test(testing);
  }

  return 0;
}
