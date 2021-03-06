#include "nGram.hpp"

NgramModel::NgramModel()
{
  idCounter = 1;
  phraseDelimiters = "\".?!#;:)(";  // octothorpe is user defined
  rawDelimiters = "\"?!#;:)(, "; //all but period
  wordDelimiters = ", ";
  delimiters = phraseDelimiters;  //other special chars? could be useful for technical texts, eg, financial reports
  delimiters += wordDelimiters;
  //delimiters += "'";
  wordDelimiter = ' ';
  phraseDelimiter = '#';

  for(int i = 0; i < NLAMBDAS; i++){
    lambdas.l[i] = 1.0;
  }

  lambdas.l[1] = 0.05;
  lambdas.l[2] = 0.3;
  lambdas.l[3] = 0.4;
  lambdas.l[4] = 0.2;
}

NgramModel::~NgramModel()
{

  //none of this is necessary, since the destructors of these will be called anyway when the main object goes out of scope...
  KeyStringTable.clear();
  StringKeyTable.clear();

  unigramTable.clear();
  bigramTable.clear();
  trigramTable.clear();
  quadgramTable.clear();
}

void NgramModel::WordToKeySequence(vector<string>& wordVec, vector<IntKey>& keySequence)
{
  int i;
  IntKey wordKey;

  //convert the string/word sequence to a sequence of integer keys
  for(i = 0; i < wordVec.size()-NGRAM-1; i++){
    wordKey = StringToKey(wordVec[i]);
    keySequence.push_back(wordKey);
  }
  wordVec.clear();
}

//prunes words of frequency<=1 from some very long sequence of words. Typically used
//to reduce the number of keys that need to be stored (eg, to fit al keys in U16, for fewer than 65k unique words, thereby allowing 64bit 4-gram table keys)
void NgramModel::PruneSequence(vector<string>& wordVec)
{
  U32 i, counter1, counter2;
  map<string,U32> freqMap;
  map<string,U32>::iterator it;
  vector<string> tempVec;

  //cout << "wordVec.size()=" << wordVec.size() << endl;
  cout << "Beginning low-frequency term (<= 1 count) pruning..." << endl;
  for(i = 0; i < wordVec.size(); i++){
    it = freqMap.find(wordVec[i]);
    if(it == freqMap.end()){
      freqMap[wordVec[i]] = 1;
    }
    else{
      freqMap[wordVec[i]]++;
    }
  }
  counter1 = counter2 = 0;
  for(it = freqMap.begin(); it != freqMap.end(); ++it){
    if(it->second <= 1){
      counter1++;
    }
    if(it->second <= 2){
      counter2++;
    }
  }
  //cout << "Nunique=" << freqMap.size() << "  Nelements<1=" << counter1 << "  Nelements<2=" << counter2 << endl;

  //now filter infrequent terms
  for(i = 0; i < wordVec.size(); i++){
    if(freqMap[wordVec[i]] > 1){
      tempVec.push_back(wordVec[i]);
    }
  }
  wordVec.clear();
  for(i = 0; i < tempVec.size(); i++){
    wordVec.push_back(tempVec[i]);
  }
  cout << "Prune completed. " << counter1 << " elements of " << freqMap.size() << " unique elements eliminated, for " << (freqMap.size()-counter1) << " keys" << endl;
  //cout << "done. wordVec.size()=" << wordVec.size() << endl;
}

void NgramModel::Train(const string& fname)
{
  int i;
  IntKey wordKey;
  U64 bigramKey, trigramKey, quadgramKey;
  vector<string> wordVec;
  vector<IntKey> keySequence;

  TextToWordSequence(fname,wordVec);
  PruneSequence(wordVec);  //very brutish, but see header. Drops very unlikely terms (freuency==1) from the sequence, freeing many int-keys
  WordToKeySequence(wordVec,keySequence);

  cout << "sequence build complete. keySequence.size()=" << keySequence.size() << " KeyStringTable.size()=" << KeyStringTable.size() << " StringKeyTable.size()=" << StringKeyTable.size() << endl;
  cout << "Building n-gram models..." << endl;

  //build the models, based on the integer key sequence
  for(i = 0; i < keySequence.size()-NGRAM-1; i++){
    //uni
    UpdateUnigramModel(unigramTable,keySequence[i]);

    //bi
    bigramKey = MakeNgramModelKey(2,keySequence[i]);
    UpdateNgramModel(bigramTable,bigramKey,keySequence[i+1]);

    //tri
    trigramKey = MakeNgramModelKey(3,keySequence[i],keySequence[i+1]);
    UpdateNgramModel(trigramTable,trigramKey,keySequence[i+2]);

    //quad
    quadgramKey = MakeNgramModelKey(4,keySequence[i],keySequence[i+1],keySequence[i+2]);
    UpdateNgramModel(quadgramTable,quadgramKey,keySequence[i+3]);

    if(i % 10000 == 9999){
      cout << "\r" << ((double)(i * 100) / (double)keySequence.size()) << "% complete        " << flush;
    }
  }
  cout << "\nN-gram model training completed, processing tables..." << endl;

  //converts all tables to conditional log-probability space. This means lower values (logs) are more likely, which can be problematic
  //for linear interpolation, which sums estimates from multiple models: if a model returns no value (zero), then it boosts
  //that particular prediction's value by having the effect of lowering the sum.
  //TablesToLogSpace();
  NormalizeTables();
  cout << "Processing complete." << endl;

  cout << "Beginning lambda expectation-maximization..." << endl;
  LambdaEM();
}

U64 NgramModel::MakeNgramModelKey(int model, IntKey w1, IntKey w2, IntKey w3)
{
  U64 ret = 0;

  switch(model){
    case 1:
    case 2:
        ret = 0x000000FF & (U64)w1;
      break;
    case 3:
        ret = ((U64)w1 << 16) | (U64)w2;
        ret &= 0x0000FFFF;
      break;
    case 4:
        ret = ((U64)w1 << 32) | ((U64)w2 << 16) | (U64)w3;
        ret &= 0x00FFFFFF;
      break;
    default:
        cout << "ERROR model# " << model << " not found in BuildNgramModelKey()" << endl;
      break;
  }

  return ret;
}

//a special case, since the unigram model only tracks, well, unigrams. There are no subkeys, the primary keys are stored redundantly as subkeys
void NgramModel::UpdateNgramModel(NgramTable& table, U64 key, IntKey nextWord)
{
  OuterTableIt it = table.find((U64)key);

  if(it == table.end()){
    table[key][nextWord] = 1;
  }
  else{
    table[key][nextWord]++;
  }
}

//a special case, since the unigram model only tracks, well, unigrams. There are no subkeys, the primary keys are stored redundantly as subkeys
void NgramModel::UpdateUnigramModel(NgramTable& unigrams, IntKey key)
{
  OuterTableIt it = unigrams.find((U64)key);

  if(it == unigrams.end()){
    unigrams[(U64)key][key] = 1;
  }
  else{
    unigrams[(U64)key][key]++;
  }
}

void NgramModel::Test(const string& fname)
{
  int i;
  vector<string> wordVec;
  vector<IntKey> keySequence;
  ResultList results;

  TextToWordSequence(fname,wordVec);
  WordToKeySequence(wordVec,keySequence);

  for(i = 0; i < keySequence.size()-NGRAM-1; i++){
    Predict(keySequence,i,results);
    ScoreResult(keySequence[i], results);
    results.clear();

    if(i % 100 == 99){
      PrintResults();
    }
  }
}

void NgramModel::PrintResults(void)
{
  cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
  cout << "nPredictions: " << lambdas.nPredictions << endl;
  cout << "recall: " << (100 * (lambdas.recall / lambdas.nPredictions)) << "%" << endl;
  cout << "bool accuracy: " << (100 * (lambdas.boolAccuracy / lambdas.nPredictions)) << "%" << endl;
  cout << "real accuracy: " << (100 * (lambdas.realAccuracy / lambdas.nPredictions)) << "%" << endl;
  cout << "top7 accuracy: " << (100 * (lambdas.topSevenAccuracy / lambdas.nPredictions)) << "%" << endl;
}

//converts raw integer frequency counts to direct conditional probabilities (or likelihoods)
void NgramModel::NormalizeTables(void)
{
  NormalizeUnigramTable(unigramTable);
  NormalizeTable(bigramTable);
  NormalizeTable(trigramTable);
  NormalizeTable(quadgramTable);
}

//a special case, since the unigram table's structure is a little different
void NgramModel::NormalizeUnigramTable(NgramTable& unitable)
{
  double sum;
  OuterTableIt outer;

  sum = 0.0;
  for(outer = unitable.begin(); outer != unitable.end(); ++outer){  
    sum += outer->second.begin()->second;
  }

  if(sum > 0.0){
    //normalize all the relative probs
    for(outer = unitable.begin(); outer != unitable.end(); ++outer){  
      outer->second.begin()->second /= sum;
    }
  }
  else{
    cout << "ERROR div zero attempted in UnigramTableToCondProbs" << endl;
  }
}

//Converts a table of raw frequency counts to conditional probability entries
void NgramModel::NormalizeTable(NgramTable& table)
{
  double sum;
  OuterTableIt outer;
  InnerTableIt inner;

  for(outer = table.begin(); outer != table.end(); ++outer){
    sum = 0.0;
    for(inner = outer->second.begin(); inner != outer->second.end(); ++inner){
      sum += inner->second;
    }

    //div zero check
    if(sum > 0.0){
      //normalize this subset of vals
      for(inner = outer->second.begin(); inner != outer->second.end(); ++inner){
        inner->second /= sum;
      }
    }
    else{
      cout << "ERROR div zero attempted in TableToCondProbs" << endl;
    }
  }
}

void NgramModel::TableToLogSpace(NgramTable& table)
{
  OuterTableIt outer;
  InnerTableIt inner;
  double sum;

  for(outer = table.begin(); outer != table.end(); ++outer){
    //get the sum for this subset
    sum = 0.0;
    for(inner = outer->second.begin(); inner != outer->second.end(); ++inner){
      sum += inner->second;
    }

    //div zero check
    if(sum <= 0.0){
      cout << "ERROR div zero caught in TableToLogProb. sum=" << endl;
      return;
    }

    //convert each entry to a (conditional) log probability
    for(inner = outer->second.begin(); inner != outer->second.end(); ++inner){
      inner->second = -1.0 * log2(inner->second/sum);
    }
  }
}

//an exception case wrt the previous function, since the unigram model structure is unique
void NgramModel::UnigramTableToLogSpace(NgramTable& unigrams)
{
  OuterTableIt outer;
  InnerTableIt inner;
  double sum = 0.0;

  for(outer = unigrams.begin(); outer != unigrams.end(); ++outer){
    sum += outer->second.begin()->second;
  }
  
  //div zero check
  if(sum <= 0.0){
    cout << "ERROR div zero caught in UnigramTableToLogSpace. sum=" << endl;
    return;
  }

  for(outer = unigrams.begin(); outer != unigrams.end(); ++outer){
    outer->second.begin()->second = -1.0 * log2(outer->second.begin()->second / sum);
  }
}

//converts tables to a log-probability, to help offset underflow risks
void NgramModel::TablesToLogSpace(void)
{
  TableToLogSpace(unigramTable);
  TableToLogSpace(bigramTable);
  TableToLogSpace(trigramTable);
  TableToLogSpace(quadgramTable);
}

bool byLogProb(const ResultPair& left, const ResultPair& right)
{
  return left.second < right.second;
}

bool byRealProb(const ResultPair& left, const ResultPair& right)
{
  return left.second > right.second;
}

//predicts based on linear interpolation over 1, 2, 3, and 4-gram log probabilities.
//uses simple smoothing, but nothing fancy.
//Since the models were all trained in the same data, the 4-gram model can be used
//to project the results across the lesser models, but this is not valid otherwise.
//Thus, look up the 4-gram result set; then for each of these, sum across the lesser model values.
void NgramModel::Predict(vector<IntKey> keySeq, int i, ResultList& results)
{
  double min3, min4;
  U64 key4g, key3g, key2g, key1g;
  OuterTableIt outer;
  InnerTableIt inner;
  unordered_set<IntKey> dupeSet;

  if(i < 3){ //index check
    return;
  }

  //get all the keys
  key4g = MakeNgramModelKey(4, keySeq[i-3], keySeq[i-2], keySeq[i-1]);
  key3g = MakeNgramModelKey(3, keySeq[i-2], keySeq[i-1]);
  key2g = MakeNgramModelKey(2, keySeq[i-1]);

  // (very) simple smoothing parameters for missing data
  min3 = min4 = 99999;

  //interpolate over 4 grams
  outer = quadgramTable.find(key4g);
  if(outer != quadgramTable.end()){
    for(inner = outer->second.begin(); inner != outer->second.end(); ++inner){
      //Add all four-gram results to the dupe set, so we don't re-estimate these words for the 3- and 2-gram queries
      //We can add "all" only because this is the first model being queried, and it contains no duplicate
      dupeSet.insert(inner->second);

      ResultPair result(inner,0);
      result.second  = lambdas.l[1] * GetProb(1,(U64)inner->first,inner->first);
      result.second += lambdas.l[2] * GetProb(2,key2g,inner->first);
      result.second += lambdas.l[3] * GetProb(3,key3g,inner->first);
      result.second += lambdas.l[4] * inner->second;
      results.push_back(result);
      if(inner->second < min4){
        min4 = inner->second;
      }
    }
  }
  else{
    min4 = 0.0;
    //min4 = 2 / (quadgramTable.size() + 1);  //laplace smooothing, if no results
  }

  //add 3 gram model results
  outer = trigramTable.find(key3g);
  if(outer != trigramTable.end()){
    for(inner = outer->second.begin(); inner != outer->second.end(); ++inner){
      if(dupeSet.count(inner->second) == 0){
        dupeSet.insert(inner->second);
        ResultPair result(inner,0);
        result.second  = lambdas.l[1] * GetProb(1,(U64)inner->first,inner->first);
        result.second += lambdas.l[2] * GetProb(2,key2g,inner->first);
        result.second += lambdas.l[3] * inner->second;
        result.second += lambdas.l[4] * min4;  //smooth missing data by the minimal four-gram estimate 
        results.push_back(result);
        if(inner->second < min3){
          min3 = inner->second;
        }
      }
    }
  }
  else{
    min3 = 0.0;
    //min3 = 2 / (trigramTable.size() + 1);
  }

  //add the 2 gram results
  outer = bigramTable.find(key2g);
  if(outer != bigramTable.end()){
    for(inner = outer->second.begin(); inner != outer->second.end(); ++inner){
      if(dupeSet.count(inner->second) == 0){
        dupeSet.insert(inner->second);
        ResultPair result(inner,0);
        result.second  = lambdas.l[1] * GetProb(1,(U64)inner->first,inner->first);
        result.second += lambdas.l[2] * inner->second;
        result.second += lambdas.l[3] * min3;  //smooth both the missing four gram and three gram data
        result.second += lambdas.l[4] * min4;
        results.push_back(result);
      }
    }
  }

  //lastly sort the results
  if(results.size() > 0){
    results.sort(byRealProb);
  }

  /*
  //dbg
  string temp;
  KeyToString(keySeq[i-3],temp);
  cout << "first 50 results for >" << temp << " ";
  KeyToString(keySeq[i-2],temp);
  cout << temp << " ";
  KeyToString(keySeq[i-1],temp);
  cout << temp << endl;

  int j = 0;
  for(ResultListIt it = results.begin(); it != results.end() && j < 50; ++it, j++){
    if(KeyToString(it->first->first,temp)){
      cout << j << ": <" << temp << "|" << it->second << ">" << endl;
    }
    else{
      cout << "shouldn't hit this line???" << endl;
    }
  }
  */
}

/*
  Train the lambdas to maximize their interpolated value over an aggregate of predictions.
  For mixture models, it can be shown that at least in general, an approximation of the best lambda
  values can be found by setting each one to the accuracy of that model, normalized to one.
  For instance, if model1 has next-word predictions with accuracy 0.3, and model2 has accuracy 0.5,
  then the appropriate lambsas are 0.3 / (0.3 + 0.5) and 0.5 / (0.3 + 0.5). This can then be extended
  to multiple models. Thus, the expected value of each model determines its appropriate weight.

  This can be written given this simplification, and then possibly improved using an iterative search procedure
  to search for more ideal lambda values.

  This is my own EM method, which automates the search for the maximal lambdas for only four models. A lot of
  it is hard-coded, and would be difficult to extend due to the mathematical complexity of EM. It 
  is very similar to Branch and Bound tasks of searching through the parameter space for the optimal parameter set.
*/
void NgramModel::LambdaEM(void)
{
  int i;
  U64 key;
  double biCt, triCt, quadCt, normal;
  vector<string> wordVec;
  vector<U16> keySeq;

  string fname = "../../oanc_SlateLambdaTraining.txt";
  TextToWordSequence(fname,wordVec);
  WordToKeySequence(wordVec,keySeq);
  wordVec.clear();

  biCt = triCt = quadCt = 0.0;

  cout << "Calculating bigram model precision..." << endl;
  //get expected value of bigram model predictions
  for(i = NGRAM + 1; i < (keySeq.size() - NGRAM - 1); i++){
    key = MakeNgramModelKey(2,keySeq[i]); // IntKey w1, IntKey w2 = 0, IntKey w3 = 0);
    //track only boolean accuracy. Check if max prediction exactly matches next word.
    if(keySeq[i+1] == GetMax(bigramTable,key)){
      biCt++;
    }
  }
  biCt = biCt / (double)i;

  cout << "trigram model.size()=" << trigramTable.size() << endl;
  cout << "Done. Calculating trigram model precision..." << endl;
  //get expected value of trigram model predictions
  for(i = NGRAM + 1; i < (keySeq.size() - NGRAM - 1); i++){
    key = MakeNgramModelKey(3,keySeq[i-1], keySeq[i]); // IntKey w1, IntKey w2 = 0, IntKey w3 = 0);
    //track only boolean accuracy. Check if max prediction exactly matches next word.
    if(keySeq[i+1] == GetMax(trigramTable,key)){
      triCt++;
    }
  }
  triCt = triCt / (double)i;

  cout << "qgram model.size()=" << quadgramTable.size() << endl;
  cout << "Done. Calculating quadgram model precision..." << endl;
  //get expected value of quadgram model predictions
  for(i = NGRAM + 1; i < (keySeq.size() - NGRAM - 1); i++){
    key = MakeNgramModelKey(4,keySeq[i-2],keySeq[i-1],keySeq[i]); // IntKey w1, IntKey w2 = 0, IntKey w3 = 0);
    //track only boolean accuracy. Check if max prediction exactly matches next word.
    if(keySeq[i+1] == GetMax(quadgramTable,key)){
      quadCt++;
    }
  }
  quadCt = quadCt / (double)i;

  normal = biCt + triCt + quadCt;
  lambdas.l[1] = biCt / (normal * 2);
  lambdas.l[2] = biCt / normal;
  lambdas.l[3] = triCt / normal;
  lambdas.l[4] = quadCt / normal;


  cout << "Model precision (uni, bi, tri, quad), per " << (keySeq.size()-NGRAM-1) << " held-out predictions: ";
  cout << lambdas.l[1] << " " << lambdas.l[2] << " " << lambdas.l[3] << " " << lambdas.l[4] << endl;

/*
  //Lastly, do the refined search procedure
  //start by finding the ordering of the optimization: order the models to be optimized, with the most accurate one first
  int maxLambda, numModel, iteration;
  bool found;
  ResultList results;
  ResultListIt it;
  double rank, maxScore, delta;
  double realScores[40];
  double curLambdas[40];

  //for each model, increment and decrement the lambda. Fix if there is improvement
  //for each lambda, run the 10k or so predictions for 10 or so lambda configurations
  for(numModel = 1; numModel <= 4; numModel++){ //start with 4-gram model and work back, since 4-gram tends to be most precise

    //reset scores, deploy new lambda set for testing, and afterward evaluate which value performed the best
    for(i = 0; i < 40; i++){
      realScores[i] = 0.0;
    }
    for(i = 0, delta = -0.40; i < 40; i++, (delta += 0.02)){
      curLambdas[i] = lambdas.l[numModel] + delta;
    }

    //iterate n times, storing the score for each run, where each run gets a slightly different value for the current lambda
    for(iteration = 0; iteration < 40; iteration++){
      if(curLambdas[iteration] >= 0.0){ //only test if lambda is >= 0
        lambdas.l[numModel] = curLambdas[iteration];
        for(i = NGRAM + 1; i < (keySeq.size() - NGRAM - 1) && i < 100; i++){
          Predict(keySeq,i,results);
          //score each result by real score only, which gives a decent real-value of method accuracy
          for(rank = 0.0, found = false, it = results.begin(); !found && it != results.end(); rank++, ++it){
            if(it->first->first == keySeq[i+1]){
              realScores[iteration] += (1.0 - (rank / (double)results.size()));
              found = true;
            }
          }
          results.clear();
          if(i % 200 == 199){
            cout << "\r" << numModel << ":" << iteration << " real " << ((realScores[iteration] / (double)(i-NGRAM)) * 100) << "%    " << flush;
          }
        }
        realScores[iteration] /= (double)(i-NGRAM);
      }
      else{
        realScores[iteration] = -1.0; //set as value such that findMax procedure never selects this lambda
      }
    }

    //take the max of the scores and fix as this lambda
    maxScore = -1.0;
    cout << "lambda scores, nmodel " << nModel << ":" << endl;
    for(i = 0; i < 40; i++){
      cout << realScores[i] << ":" << curLambdas[i] << endl;
      if(realScores[i] > maxScore){
        maxScore = realScores[i];
        maxLambda = i;
      }
    }

    //fix the current lambda as the max per this test
    lambdas.l[numModel] = curLambdas[maxLambda]; 
  }

  normal = 0.0;
  for(i = 1; i <= 4; i++){
    normal += lambdas.l[i]; 
  }
  for(i = 1; i <= 4; i++){
    lambdas.l[i] /= normal;
  }

  cout << "final lambdas: " << endl;
  for(i = 1; i <= 4; i++){
    cout << i << ": " << lambdas.l[i] << endl; 
  }
*/
}


//Small utility for immediately returning only the most likely word prediction, given some key (representing the preceding word sequence)
U16 NgramModel::GetMax(NgramTable& table, U64 outerKey)
{
  double max;
  U16 ret = 0;
  OuterTableIt outer = table.find(outerKey);

  //key exists, so find the max-likely word within the subset
  if(outer != table.end()){
    max = 0.0;
    for(InnerTableIt inner = outer->second.begin(); inner != outer->second.end(); ++inner){
      if(inner->second > max){
        ret = inner->first;
        max = inner->second;
      }
    }
  }

  return ret;
}

//Handles table probability lookups, given complete U64/U16 key/subkey. Returns prob if found, else returns 0.0
double NgramModel::GetProb(int nModel, U64 key, U16 subkey)
{
  double ret;
  OuterTableIt outer;
  InnerTableIt inner;
  
  ret = 0.0;  //return 0.0 by default
  switch(nModel){
    case 1:
      outer = unigramTable.find(key);
      if(outer != unigramTable.end()){
        inner = outer->second.find(subkey);
        if(inner != outer->second.end()){
          ret = inner->second;
        }
      }
      break;
    case 2:
      outer = bigramTable.find(key);
      if(outer != bigramTable.end()){
        inner = outer->second.find(subkey);
        if(inner != outer->second.end()){
          ret = inner->second;
        }
      }
      break;
    case 3:
      outer = trigramTable.find(key);
      if(outer != trigramTable.end()){
        inner = outer->second.find(subkey);
        if(inner != outer->second.end()){
          ret = inner->second;
        }
      }
      break;
    case 4:
      outer = quadgramTable.find(key);
      if(outer != quadgramTable.end()){
        inner = outer->second.find(subkey);
        if(inner != outer->second.end()){
          ret = inner->second;
        }
      }
      break;
    default:
      cout << "ERROR model " << nModel << " not found in GetProb" << endl;
  }

  return ret;
}

void NgramModel::ScoreResult(IntKey actual, ResultList& results)
{
  double i;
  ResultListIt it;

  lambdas.nPredictions++;

  if(actual == results.begin()->first->first){
    lambdas.boolAccuracy++;
  }

  i = 1.0;
  for(it = results.begin(); it != results.end(); ++it, i++){
    if(it->first->first == actual){
      lambdas.recall++;
      lambdas.realAccuracy += (1 / i);
      if(i <= 7){
        lambdas.topSevenAccuracy++;
      }
    }
  }
}

//returns IntKey key of a word, or allocates a new key for the word if it doesn't already exist
IntKey NgramModel::StringToKey(const string& word)
{
  IntKey ret;
  StringKeyMapIt it = StringKeyTable.find(word);

  if(it != StringKeyTable.end()){
    ret = it->second;
  }
  else{
    if(!AllocKey(word,ret)){
      cout << "ERROR could not alloc new key in StringToKey" << endl;
      ret = 0;
    }
  }

  return ret;
}

bool NgramModel::KeyToString(IntKey key, string& str)
{
  bool ret;
  KeyStringMapIt it = KeyStringTable.find(key);

  if(it != KeyStringTable.end()){
    str = it->second;
    ret = true;
  }
  else{
    cout << "ERROR key " << key << " not found in KeyStringTable." << endl;
    ret = false;
  }

  return ret;
}

//alloc a new id for some string. If word is not new, returns the existing key.
bool NgramModel::AllocKey(const string& newWord, IntKey& key)
{
  bool ret = false;
  StringKeyMapIt it = StringKeyTable.find(newWord);  

  if(idCounter >= U32_MAX){
    cout << "ERROR out of IntKey keys for new words!" << endl;
    cout << "StringKeyTable.size()=" << StringKeyTable.size() << " KeyStringTable.size()=" << KeyStringTable.size() << endl;
    ret = false;
  }
  //alloc new id if no key exists for this string
  else if(it == StringKeyTable.end()){
    StringKeyTable[newWord] = idCounter;
    KeyStringTable[idCounter] = newWord;
    key = idCounter;
    idCounter++;
    ret = true;
  }
  // else string already has a key, so just return the existing one
  else{
    key = it->second;
    ret = true;
  }

  return ret;
}

/*
  Do our best to clean the sample. We try to preserve as much of the author's style as possible,
  so for instance, we don't expand contractions, viewing them instead as synonyms. "Aren't" and
  "are not" thus mean two different things.

  ASAP change params to both string
*/
void NgramModel::NormalizeText(char ibuf[BUFSIZE], string& ostr)
{
  string istr = ibuf;
  //string ostr;

  /*
  cout << len << " (first 120ch) length input sample looks like: (chct=" << len << "," << strlen(buf) << "):" << endl;
  for(i = 0; i < 120; i++){
    putchar(buf[i]);
  }
  cout << "<end>" << endl;
  */

  //filter pipeline
  //TODO: map abbreviations.
  //mapAbbreviations(istr,ostr);  //demark Mr. Ms., other common abbreviations before rawPass() uses periods as phrase delimiters
  //post: "Mr." will instead be "Mr+". We'll use '+' to convert back to "Mr." after calling delimitText() 
  //mapContractions(obuf.c_str(), obuf2);
  //dumb for now. obuf will be used later for more context-driven text preprocessing

  //filters and context-free transformers
  RawPass(istr);
  //cout << "1: " << buf << endl;
  ToLower(istr);
  //cout << "2: " << buf << endl;
  ScrubHyphens(istr);
  //cout << "3: " << buf << endl;
  DelimitText(istr);
  //cout << "4: " << buf << endl;
  FinalPass(istr);

  ostr = istr; //note this effectively clears any previous contents of ostr, which is intended

  //cout << "here: " << buf[0] <<  endl;
  /*
  cout << "120ch output sample looks like: " << endl;
  for(i = 0; i < 120; i++){
    putchar(buf[i]);
  }
  cout << "<end>" << endl;
  cin >> i;
  */
}

bool NgramModel::IsPhraseDelimiter(char c)
{
  U32 i;

  /*
  if(c == 34){
    return true;
  }
  */

  for(i = 0; (phraseDelimiters[i] != '\0') && (i < phraseDelimiters.length()); i++){
    if(c == phraseDelimiters[i]){
      return true;
    }
  }

  return false;
}

/*
  Replaces delimiter chars with either phrase (#) or word (@) delimiters.
  This is brutish, so input string must already be preprocessed.
  Output can be used to tokenize phrase structures and words.


  Notes: This function could be made more advanced. Currently it makes direct replacement
  of phrase/word delimiters with our delimiters (without regard to context, etc)
*/
void NgramModel::DelimitText(string& istr)
{
  int i, k;

  for(i = 0; istr[i] != '\0'; i++){
    if(IsPhraseDelimiter(istr[i])){  //chop phrase structures
      istr[i] = phraseDelimiter;
      
      //consume white space and any other delimiters (both phrase and words delims):  "to the park .  Today" --becomes--> "to the park####Today"
      k = i+1;
      while((istr[k] != '\0') && IsDelimiter(istr[k],this->delimiters)){
        istr[k] = phraseDelimiter;
        k++;
      }
    }
    else if(IsWordDelimiter(istr[i])){
      istr[i] = wordDelimiter;

      //consume right delimiters
      k = i+1;
      while((istr[k] != '\0') && IsWordDelimiter(istr[k])){
        istr[k] = wordDelimiter;
        k++;
      }
    }
  }
}

bool NgramModel::IsWordDelimiter(char c)
{
  int i;

  for(i = 0; (wordDelimiters[i] != '\0') && (i < wordDelimiters.length()); i++){
    if(c == wordDelimiters[i]){
      return true;
    }
  }

  return false;
}

bool NgramModel::IsValidWord(const char* word)
{
  string w = word;
  return IsValidWord(w);
}
bool NgramModel::IsValidWord(const string& token)
{
  //no words longer than limit
  if(token.length() > MAX_WORD_LEN){
    //cout << "\rWARN unusual length word in isValidWord: >" << token << "< ignored. Check parsing                        " << endl;
    return false;
  }

  if(token[0] == '\''){  // filters much slang: 'ole, 'll, 'em, etc
    return false;
  }
  if((token[0] == '*') || (token[1] == '*')){
    return false;
  }
  if(token == "com"){  //lots of "www" and ".com" etc
    //cout << "invalid word: " << token << endl;
    return false;
  }
  if(token == "www"){  //lots of "www"
    //cout << "invalid word: " << token << endl;
    return false;
  }
  if(token == "http"){  //lots of "http"
    //cout << "invalid word: " << token << endl;
    return false;
  }

  if((token[0] == '\'') && ((token[1] == '\'') || (token[1] == 's'))){  //hack: covers '' and 's in output stream
    return false;
  }
  if(token == "th"){  //occurs when "8th" is converted to "th" after numeric drop
    return false;
  }

  for(U32 i = 0; i < token.length(); i++){
    if((token[i] >= 47) && (token[i] <= 64)){ ///exclude all of 0123456789@:;<=>?
      //cout << "invalid word: " << token << endl;
      return false;
    }
    if((token[i] >= 35) && (token[i] <= 38)){ ///exclude all of #$%&
      //cout << "invalid word: " << token << endl;
      return false;
    }
    if((token[i] >= 91) && (token[i] <= 96)){ ///exclude all of []^`_
      //cout << "invalid word: " << token << endl;
      return false;
    }
  }

  return true;
}


/*
  Hyphens are ambiguous since they can represent nested phrases or compound words:
    1) "Mary went to the park--ignoring the weather--last Saturday."
    2) "John found a turquoise-green penny."
  Process string by checking for hyphens. Double hyphens represent nested
  phrases, so will be changed to phrase delimiter. Single hyphens will
  be changed to a word-delimiter (a space: ' ').

  Notes: this function snubs context. Huck Finn contains a lot of hyphens, without
  much regular structure (except double vs. single). More advanced parsing will be needed
  to preserve nested context: <phrase><hyphen-phrase><phrase>. Here the first and second <phrase>
  are severed contextually if we just shove in a delimiter. Re-organizing the string would be nice,
  but also difficult for parses like <phrase><hyphen-phrase><hyphen-phrase><hyphen-phrase><phrase> 
  or <phrase><hyphen-phrase><phrase><hyphen-phrase><phrase> which occur often in Huck Finn.
*/
void NgramModel::ScrubHyphens(string& istr)
{
  int i;

  for(i = 0; istr[i] != '\0'; i++){
    if((istr[i] == '-') && (istr[i+1] == '-')){  //this snubs nested context
      istr[i+1] = istr[i] = phraseDelimiter;
    }
    else if(istr[i] == '-'){   //this could also use more context sensitivity: eg, "n-gram" should not necessarily resolve to "n" and "gram" since n is not a word
      istr[i] = wordDelimiter;
    }
  }
}

/*
  Convert various temp tags back to their natural equivalents.
  For now, just converts "Mr+" back to the abbreviation "Mr."
*/
void NgramModel::FinalPass(string& buf)
{
  for(int i = 0; i < buf.length(); i++){
    if(buf[i] == PERIOD_HOLDER){
      buf[i] = '.';
    }
  }
}

void NgramModel::ToLower(string& myStr)
{
  for(int i = 0; i < myStr.length(); i++){
    if((myStr[i] >= 'A') && (myStr[i] <= 'Z')){
      myStr[i] += 32;
    }
  }
}

//standardize input by converting to lowercase
void NgramModel::ToLower(char buf[BUFSIZE])
{
  int i;

  for(i = 0; buf[i] != '\0'; i++){
    if((buf[i] >= 'A') && (buf[i] <= 'Z')){
      buf[i] += 32;
      //cout << buf[i] << " from " << (buf[i] -32) << endl;
    }
  }
}

/*
  Raw char transformer. currently just replaces any newlines or tabs with spaces. And erases "CHAPTER" headings.
  changes commas to wordDelimiter
*/
void NgramModel::RawPass(string& istr)
{
  int i;

  for(i = 0; istr[i] != '\0'; i++){
    if((istr[i] < 32) || (istr[i] > 122)){ // convert whitespace chars and extended range chars to spaces
    //if((istr[i] == '\n') || (istr[i] == '\t') || (istr[i] == '\r') || (istr[i] == ',')){
      istr[i] = wordDelimiter;
    }
    else if(istr[i] == ','){   //HACK
      istr[i] = wordDelimiter;
    }

    /*
    //HACK erase "CHAPTER n." from input, a common header in Hucklberry Finn
    // left two checks short-circuit to only allow call strncmp if next two letters are "CH" (ie, high prob. of "CHAPTER")
    if(((i + 16) < len) && (istr[i] == 'C') && (istr[i+1] == 'H') && !strncmp("CHAPTER",&istr[i],7)){
      j = i + 8;
      for( ; (i < j) && (i < len); i++){
        istr[i] = phraseDelimiter;
      }
      //now we point at 'X' in "CHAPTER X", so consume chapter numerals until we hit the period
      for( ; (istr[i] != '.') && (i < len); i++){
        istr[i] = phraseDelimiter;
      }
      if(i == len){
        cout << "ERROR istrfer misalignment in rawPass()" << endl;
      }
      else{
        istr[i++] = phraseDelimiter;  
      }
    }
    */
  }
}

/*
  This is the most general is-delim check:
  Detects if char is ANY of our delimiters (phrase, word, or other/user-defined.)
*/
bool NgramModel::IsDelimiter(const char c, const string& delims)
{
  int i;

  for(i = 0; i < delims.length(); i++){
    if(c == delims[i]){
      return true;
    }
  }

  return false;
}

void NgramModel::TextToWordSequence(const string& fname, vector<string>& wordVec)
{
  int nTokens, i;
  U32 wordCt;
  char buf[BUFSIZE];
  char* toks[MAX_TOKENS_PER_READ];
  long double fsize, progress;
  string line, word, s;
  fstream infile(fname.c_str(), ios::in);

  //stopfile.open(stopWordFile.c_str(), ios::read);
  if(!infile){
    cout << "ERROR could not open file: " << fname << endl;
    return;
  }

  //gets the file size
  fsize = (long double)infile.tellg();
  infile.seekg(0, std::ios::end);
  fsize = (long double)infile.tellg() - fsize;
  infile.seekg(0, infile.beg);

  wordVec.reserve(1 << 24); //reserve space for about 1.6 million words

  //cout << "max_size of list<string>: " << wordSequence.max_size() << "  fsize: " << fsize << endl; 

  wordCt = 0;
  while(infile.getline(buf,BUFSIZE)){  // same as: while (getline( myfile, line ).good())
    if(strnlen(buf,BUFSIZE) > 5){  //ignore lines of less than 10 chars
      //strncpy(buf,line.c_str(),255);
      buf[BUFSIZE-1] = '\0';
      s.clear();
      NormalizeText(buf,s);
      strncpy(buf,s.c_str(),BUFSIZE-1);
      buf[BUFSIZE-1] = '\0';
      nTokens = Tokenize(toks,buf,delimiters);

      //push each of these tokens to back of vector
      for(i = 0; i < nTokens; i++){
        word = toks[i];

        //no filtering except some basic validity checks
        if(IsValidWord(word)){
          word = toks[i];
          wordVec.push_back(word);
          wordCt++;
          //cout << word << " " << flush;
        }

        if((wordCt % 1000) == 0){
          progress = (long double)infile.tellg();
          cout << "\r" << (int)((progress / fsize) * 100) << "% complete wordSeq.size()=" << wordVec.size() << "             " << flush;
        }
      }
    }
  }
  cout << endl;

  //NOT HERE! init numwords after we prune the vocabulary
  //this->numwords = wordSequence.size();

  /*
  i = 0;
  for(vector<string>::iterator it = wordSequence.begin(); it != wordSequence.end(); ++it){
    cout << " " << *it;
    i++;
    if(i % 20 == 0){
      cout << " " << endl;
    }
  }
  cout << "\nend of your list sir" << endl;
  */

  infile.close();
}

/*
  Logically the same as strtok: replace all 'delim' chars with null, storing beginning pointers in ptrs[]
  Input string can have delimiters at any point or multiplicity

  Pre-condition: This function continues tokenizing until it encounters '\0'. So buf must be null terminated,
  so be sure to bound each phrase with null char.

  Testing: This used to take a len parameter, but it was redundant with null checks and made the function 
  too nasty to debug for various boundary cases, causing errors.
*/
int NgramModel::Tokenize(char* ptrs[], char buf[BUFSIZE], const string& delims)
{
  int i, tokCt;
  //int dummy;

  if((buf == NULL) || (buf[0] == '\0')){
    ptrs[0] = NULL;
    cout << "WARN buf==NULL in tokenize(). delims: " << delims << endl;
    return 0;
  }
  if(delims.length() == 0){
    ptrs[0] = NULL;
    cout << "WARN delim.length()==0 in tokenize()." << endl;
    return 0;
  }

  i = 0;
  if(IsDelimiter(buf[0], delims)){
    //consume any starting delimiters then set the first token ptr
    for(i = 0; IsDelimiter(buf[i], delims) && (buf[i] != '\0'); i++);
    //cout << "1. i = " << i << endl;
  }

  if(buf[i] == '\0'){  //occurs if string is all delimiters
    if(DBG)
      cout << "buf included only delimiters in tokenize(): i=" << i << "< buf: >" << buf << "< delims: >" << delims << "<" << endl;
    ptrs[0] = NULL;
    return 0;
  }

  //assign first token
  ptrs[0] = &buf[i];
  tokCt = 1;
  while(buf[i] != '\0'){

    //cout << "tok[" << tokCt-1 << "]: " << ptrs[tokCt-1] << endl;
    //cin >> dummy;
    //advance to next delimiter
    for( ; !IsDelimiter(buf[i], delims) && (buf[i] != '\0'); i++);
    //end loop: buf[i] == delim OR buf[i]=='\0'

    //consume extra delimiters
    for( ; IsDelimiter(buf[i], delims) && (buf[i] != '\0'); i++){
      buf[i] = '\0';
    } //end loop: buf[i] != delim OR buf[i]=='\0'

    //at next substring
    if(buf[i] != '\0'){
      ptrs[tokCt] = &buf[i];
      tokCt++;
      
/*
      if(tokCt % 200 == 0){
        cout << "tokCt=" << tokCt <<  flush;
      }
*/
    }

  } //end loop: buf[i]=='\0'

  //cout << "DEBUG first/last tokens: " << ptrs[0] << "/" << ptrs[tokCt-1] << "<end>" <<  endl; 

  ptrs[tokCt] = NULL;

  return tokCt;
}
