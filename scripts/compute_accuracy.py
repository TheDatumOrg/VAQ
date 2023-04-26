import csv
import numpy as np
import sys

def computeAvg_recall(groundtruth, answer, k):
  ans = 0.0
  for i in range(len(groundtruth)):
    ct = 0
    for p in answer[i]:
      for j in range(k):
        if (p == groundtruth[i][j]):
          ct += 1
          break
    ans += float(ct)/k
  ans /= len(groundtruth)
  return ans

def computeRecallAtR(groundtruth, answer, k):
  ans = 0.0
  for i in range(len(groundtruth)):
    truenn = groundtruth[i][0]
    if (truenn in answer[i]):
      ans += 1
  ans /= len(groundtruth)
  return ans

def computeMAP(groundtruth, answer, k):
  ans = 0.0
  for i in range(len(groundtruth)):
    ap = 0.0
    for r in range(k):
      if (answer[i][r] in groundtruth[i]):
        ct = 0
        for j in range(r+1):
          if (answer[i][j] in groundtruth[i][:r+1]):
            ct += 1
        ap += float(ct)/(r+1)
    
    ans += float(ap)/k
  ans /= len(groundtruth)
  return ans

def main():
  if (len(sys.argv) < 3):
    print('need answers & groundtruth file args')
    exit(1)
  
  infile = sys.argv[1]
  gndfile = sys.argv[2]

  groundtruth = []
  answers = []
  with open(infile, 'r', newline='') as fp:
    reader = csv.reader(fp)
    for row in reader:
      groundtruth.append(row)
  
  with open(gndfile, 'r', newline='') as fp:
    reader = csv.reader(fp)
    for row in reader:
      answers.append(row)

  # sanity check
  if (len(groundtruth) != len(answers)):
    print('Sanity check failed, len(grountruth) != len(answers)')
    exit(1)
  q_len = len(groundtruth)
  for i in range(q_len):
    if (len(groundtruth[i]) != len(answers[i])):
      print('Sanity check failed, len(grountruth[{}]) != len(answers[{}])'.format(i, i))
      exit(1)
  k = len(groundtruth[0])
  print('query size:', q_len)
  print('k:', k)

  
  # compute precision
  print('Precision =', computeAvg_recall(groundtruth, answers, k))
  print('Recall@R =', computeRecallAtR(groundtruth, answers, k))
  print('MAP =', computeMAP(groundtruth, answers, k))

if __name__ == "__main__":
  main()