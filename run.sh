#!/bin/bash
echo "Exemplo da figura 1"
./main --size 60 80 --input color_list.txt --epochs 30 --seed 753
echo "É possível invocar o programa sem opções na linha de comando."
./main
echo "Semente fixa, com delta=0.95 (padrão)"
./main --seed 0
echo "Semente fixa, com delta=0.99"
./main --seed 0 --delta 0.99
echo "Semente fixa, com delta=1"
./main --seed 0 --delta 1
echo "Exemplo da figura 2"
./main --seed 6921 --delta 0.8
echo "Exomplo da figura 3"
./main --seed 1432500040
