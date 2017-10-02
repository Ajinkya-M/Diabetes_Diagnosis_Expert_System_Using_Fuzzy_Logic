import csv

def preprocessData():
    f = open("data.txt", "r")
    reader = csv.reader(f)
    count = 0
    f_w = open("preprocessed_data.csv", "w")
    writer = csv.writer(f_w)
    for row in reader:
        values = [row[7], row[1], row[2], row[4], row[5], row[6], row[8]]
        if '0' in values[0:6]:
            continue
        writer.writerow([count]+values)
        count += 1
    f_w.close()
    f.close()

    print(count)


if __name__ == '__main__':
    preprocessData();