#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace std;

unordered_map<int, int> tf;
unordered_map<int, int> df;
unordered_map<int, int> tfidf;

// http://stackoverflow.com/questions/1060648/fast-intersection-of-sets-c-vs-c-sharp
float similarity(const vector<int> &s1, const vector<int> &s2) {
  /*
   * Set intersection and set union from algorithms would be good but are
   * slower if all we're interested in is the total counts and not the elements
  */
  float inBoth = 0;
  float inEither = 0;
  auto it1 = s1.begin();
  auto it2 = s2.begin();
  while(it1 != s1.end() && it2 != s2.end()) {
    if (*it1 < *it2) {
      ++it1;
      ++inEither;
    } else if (*it2 < *it1) {
      ++it2;
      ++inEither;
    } else {
      ++inBoth;
      ++inEither;
      ++it1;
      ++it2;
    }
  }
  while (it1 != s1.end()) {
    ++it1;
    ++inEither;
  }
  while (it2 != s2.end()) {
    ++it2;
    ++inEither;
  }
  return inBoth / inEither;
}

// Could be written as a lambda function but it's used multiple times
template <class K, class V>
bool compareSecond(pair<K, V> const& l, pair<K, V> const& r) {
  return l.second > r.second;
}

// Convert a line from SVMLight multiclass format to a vector of klasses and features
pair<vector<int>, vector<int>> convertLine(string line, bool storeCounts) {
  istringstream iss(line);
  vector<string> tokens;
  // Split tokens on white space
  copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter<vector<string>>(tokens));
  //
  vector<int> klasses;
  vector<int> features;
  for (auto &token : tokens) {
    // Naive split -- if there's a : it's a feature, else it's a class
    int pos = token.find(":");
    if (pos == string::npos) {
      // Klass
      if (token.at(token.size() -1) == ',') {
        token = token.substr(0, token.size() - 1);
      }
      klasses.push_back(atoi(token.c_str()));
    } else {
      // Feature
      auto fs = token.substr(0, pos);
      auto vs = token.substr(pos + 1);
      auto f = atoi(fs.c_str());
      auto v = atoi(vs.c_str());
      if (storeCounts) {
        tf[f] += v;
        df[f] += 1;
      }
      features.push_back(f);
    }
  }
  sort(klasses.begin(), klasses.end());
  sort(features.begin(), features.end());
  return make_pair(klasses, features);
}

int main(int argc, const char *argv[]) {
  ifstream train("../data/train.csv");
  ifstream test("../data/test.csv");
  string line;
  //
  vector<vector<int>> klassIndex;
  vector<vector<int>> featureIndex;
  unordered_map<int, vector<int>> featureLookup;

  // Load the "training" data and index by features
  int i = 0;
  while (std::getline(train, line)) {
    if (i % 10000 == 0) {
      cerr << "| Up to... " << i << endl;
    }
    auto p = convertLine(line, true);
    auto klasses = p.first;
    auto features = p.second;
    for (int f : features) {
      featureLookup[f].push_back(i);
    }
    klassIndex.push_back(klasses);
    featureIndex.push_back(features);
    ++i;
  }
  // Calculate TFIDF for avoiding calculating KNN over all elements
  cerr << "Calculating TFIDF" << endl;
  for (auto p : tf) {
    int f = p.first;
    float idf = log(klassIndex.size() / ((float)df[f]));
    tfidf[f] = log(tf[f] + 1) * idf;
  }
  // "Testing"
  i = 0;
  while (std::getline(test, line)) {
    if (i % 10000 == 0) {
      cerr << "| Up to... " << i << endl;
    }
    auto p = convertLine(line, false);
    auto klasses = p.first;
    auto features = p.second;
    // Calculate scores against potentials
    // Find the K features with highest TFIDF score
    unordered_map<int, float> featureScores;
    for (int f : features) {
      featureScores[f] = tfidf[f];
    }
    vector<pair<int, float>> bestFeatures(8);
    partial_sort_copy(featureScores.begin(), featureScores.end(), bestFeatures.begin(), bestFeatures.end(), compareSecond<int, float>);
    //
    // Only get neighbours that have one of the K "best" features
    /*
    unordered_set<int> potentials;
    for (auto p : bestFeatures) {
      int f = p.first;
      copy(featureLookup[f].begin(), featureLookup[f].end(), inserter(potentials, potentials.end()));
    }
    */
    // This is faster in many cases than creating a huge set
    unordered_map<int, float> scores;
    for (auto p : bestFeatures) {
      for (int ind : featureLookup[p.first]) {
        if (scores.count(ind) == 0) {
          scores[ind] = similarity(features, featureIndex[ind]);
        }
      }
    }
    // Get the top N similar elements
    vector<pair<int, float>> top_n(100);
    partial_sort_copy(scores.begin(), scores.end(), top_n.begin(), top_n.end(), compareSecond<int, float>);
    //
    cout << "SIMILAR " << i << ",";
    for (auto p : top_n) {cout << p.first << ":" << p.second << " ";}
    cout << endl;
    //
    cout << "COUNTS " << i << ",";
    for (auto p : top_n) {cout << p.first << ":" << klassIndex[p.first].size() << " ";}
    cout << endl;
    //
    unordered_map<int, float> klassScores;
    for (int it = 0; it < min(5, (int)top_n.size()); ++it) {
      auto &p = top_n[it];
      for (auto kls : klassIndex[p.first]) {
        klassScores[kls] += p.second;
      }
    }
    //
    vector<pair<int, float>> kS(klassScores.begin(), klassScores.end());
    sort(kS.begin(), kS.end(), compareSecond<int, float>);
    cout << "ALL_KLASSES " << i << ",";
    for (auto p : kS) {cout << p.first << ":" << p.second << " ";}
    cout << endl;
    //
    vector<int> guesses;
    for (int it = 0; it < min(3, (int)kS.size()); ++it) { guesses.push_back(kS[it].first); }
    sort(guesses.begin(), guesses.end());
    cout << i << ",";
    for (auto kls : guesses) {cout << kls << " ";}
    cout << endl;
    //
    ++i;
  }

  return 0;
}
