# without tensorflow
# python 2.7
# reference https://www.kaggle.com/c/titanic/details/getting-started-with-python

import csv as csv
import numpy as np

csv_train_file_object = csv.reader(open("/home/jana/PycharmProjects/mlRelated/titanic/data/train.csv", 'rb'))
header = csv_train_file_object.next()
print header

train_passenger_class_position = 2
train_gender_position = 4
train_age_position = 5
train_sibSp_position = 6
train_survived_position = 1

train_data = []
for row in csv_train_file_object:
    if row[train_age_position] == '':
        continue;
    # print row
    train_data.append(row)
train_data = np.array(train_data)
# print train_data

train_data_count = len(train_data)
# print "train_data_count:", train_data_count

passenger_classes = np.unique(train_data[0::, train_passenger_class_position])  # 1 2 3
passenger_class_count = len(passenger_classes)  # 3

genders = np.unique(train_data[0::, train_gender_position])  # female, male
gender_count = len(genders)  # 2

# ages_sorted = np.sort(ages)
# print "ages_sorted:", ages_sorted
age_bin_size = 10
# age_ceiling = np.amax(ages)
# print "age_ceiling: ", age_ceiling
age_bin_count = (np.amax(train_data[0::, train_age_position].astype(float)) / age_bin_size).astype(int)
# print "age_bin_count:", age_bin_count

for data_index in xrange(train_data_count):
    if train_data[data_index, train_sibSp_position].astype(float) > 0:
        train_data[data_index, train_sibSp_position] = '1'
# print "sibSp:", data[0::, sibSp_position]
sibSps = ['0', '1']
sibSps_count = len(sibSps)

survival_table = np.zeros((passenger_class_count, gender_count, age_bin_count, sibSps_count))
# print survival_table

for passenger_class_index in xrange(passenger_class_count):
    for gender_index in xrange(gender_count):
        for age_bin_index in xrange(age_bin_count):
            for sibSp_index in xrange(sibSps_count):

                survived = train_data[
                    (train_data[0::, train_passenger_class_position] == passenger_classes[passenger_class_index]) &
                    (train_data[0::, train_gender_position] == genders[gender_index]) &
                    (train_data[0::, train_age_position].astype(float) <= ((age_bin_index + 1) * age_bin_size)) &
                    (train_data[0::, train_sibSp_position] == sibSps[sibSp_index]),
                    train_survived_position
                ]

                # print "passenger_class:", passenger_classes[passenger_class_index], \
                #     " gender:", genders[gender_index], \
                #     " age_range:", ((age_bin_index + 1) * age_bin_size),\
                #     " sibSp:", sibSp_index
                # print survived
                if len(survived) == 0:
                    continue
                survival_table[passenger_class_index, gender_index, age_bin_index, sibSp_index] = np.mean(
                    survived.astype(np.float))

# print survival_table

survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1

print survival_table




prediction_file_object = csv.writer(open("/home/jana/PycharmProjects/mlRelated/titanic/output/predictions.csv", "wb"))
prediction_file_object.writerow(["passengerId", "Survived"])

csv_test_file_object = csv.reader(open("/home/jana/PycharmProjects/mlRelated/titanic/data/test.csv", 'rb'))
header = csv_test_file_object.next()
print header

test_data = []
for row in csv_test_file_object:
    # print row
    test_data.append(row)
test_data = np.array(test_data)
# print test_data

test_data_count = len(test_data)
# print "test_data_count:", test_data_count

test_passenger_id = 0
test_passenger_class_position = 1
test_gender_position = 3
test_age_position = 4
test_sibSp_position = 5

for test_data_index in xrange(test_data_count):

    row = test_data[test_data_index]
    # print row

    for passenger_class_index in xrange(passenger_class_count):
        if row[test_passenger_class_position] == passenger_classes[passenger_class_index]:
            break
    print "passenger_class_index: "
    print passenger_class_index

    for gender_index in xrange(gender_count):
        if row[test_gender_position] == genders[gender_index]:
            break
    print "gender_index: "
    print gender_index

    age = test_data[test_data_index, test_age_position]
    print "age: ", age
    if age != '':
        age_bin_index = int(int(float(age)) / age_bin_size)
    else:
        age_bin_index = 1
    print "age_bin_index: "
    print age_bin_index

    sibSp = row[test_sibSp_position]
    sibSp_index = 0;
    if sibSp == '1':
        sibSp_index = 1
    print "sibSp_index: "
    print sibSp_index

    passenger_id = row[test_passenger_id]

    survived = survival_table[passenger_class_index][gender_index][age_bin_index][sibSp_index]
    if survived < 0.5:
        survived = '0'
    else:
        survived = '1'

    prediction_file_object.writerow([passenger_id, survived])