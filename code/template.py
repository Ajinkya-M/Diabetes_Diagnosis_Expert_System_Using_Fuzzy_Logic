import csv

# Defining Constants
#Age
age_low =35
age_med = (30,50)
age_high = 45
#plasma glucose concentration
glucose_low = 100
glucose_med = (90,160)
glucose_high = 150
#diabolic blood pressure
dbp_low = 70
dbp_med = (60,100)
dbp_high = 90
#Insulin
insulin_low = 150
insulin_med = (140,200)
insulin_high = 190
#Body Mass Index
bmi_low = 23
bmi_med = (23,30)
bmi_high = 27
#Diabetes Pedigree function
dpf_low = 0.4
dpf_med = (0.2,0.8)
dpf_high = 0.6


values_list = []

def fuzzySets(low, med, high, val):
    if(val < med[0]):
        return 'L'
    elif(val >= med[0] and val <low):
        a = (med[0]+low)/2
        if(val >= a):
            return 'M'
        else:
            return 'L'
    elif (val >= low and val < high):
        return 'M'
    elif (val >= high and val < med[1]):
        a = (med[1] + high) / 2
        if (val >= a):
            return 'H'
        else:
            return 'M'
    elif (val >= med[1]):
        return 'H'


def calculate_degree_of_attr(attr, value, low, med, high):
    deg = 0
    if(attr == 'L'):
        if(value >= med[0]):
            dist = abs(med[0] - value)
            deg = 1 - (dist * abs(med[0] - low)) / 100
        else:
            deg = 1
    elif(attr == 'M'):
        dist = ((med[0]+med[1]) / 2) - value
        deg = 1 - abs(dist) / 100
    elif(attr == 'H'):
        if (value <= med[1]):
            dist = abs(med[1] - value)
            deg = 1 - (dist * abs(med[1] - high)) / 100
        else:
            deg = 1

    return deg

def calculate_degree_of_rule(rule):
    global values_list
    values = values_list[rule[0]]
    deg_age = calculate_degree_of_attr(rule[1], values[1], age_low, age_med, age_high)
    deg_glucose = calculate_degree_of_attr(rule[2], values[2], glucose_low, glucose_med, glucose_high)
    deg_dbp = calculate_degree_of_attr(rule[3], values[3], dbp_low, dbp_med, dbp_high)
    deg_insulin = calculate_degree_of_attr(rule[4], values[4], insulin_low, insulin_med, insulin_high)
    deg_bmi = calculate_degree_of_attr(rule[5], values[5], bmi_low, bmi_med, bmi_high)
    deg_dpf = calculate_degree_of_attr(rule[6], values[6], dpf_low, dpf_med, dpf_high)
    deg = deg_age * deg_glucose * deg_dbp * deg_insulin * deg_bmi *deg_dpf

    return deg

def conflict_rssolve(rule1, rule2):
    if(rule1[7] == rule2[7]):
        return rule1[7]
    else:
        #resolve conflict
        deg_rule1 = calculate_degree_of_rule(rule1)
        deg_rule2 = calculate_degree_of_rule(rule2)

        if(deg_rule1 >= deg_rule2):
            return rule1[7]
        else:
            return rule2[7]


def check_conflict(rules_list):
    qualified_rules_list = []
    count = 0
    flag = 1
    for rule in rules_list:
        if(count <= len(rules_list)):
            print("Rule from rule list  :  ", rule)
            for qrule in qualified_rules_list:
                flag = 1
                print("Rule from qualified rule list  :  ", qrule)
                if(rule[1:7] == qrule[1:7]):
                    print("MATCH IDENTIFIED for ",rule[0], qrule[0])
                    qrule[7] = conflict_rssolve(rule, qrule)
                    flag = 0
                    break
            if flag == 1 :
                print(rule[0], " appended in list")
                qualified_rules_list.append(rule)

        count += 1
    with open("rules_generated.txt", "w") as f_r:
        for r in qualified_rules_list:
            f_r.write("Rule No. "+str(r[0])+" --> "+" If Age is "+str(r[1])+" AND Glucose is "+str(r[2])+
                      " AND Blood Pressure is "+str(r[3])+" AND Insulin is "+str(r[4])
                      + " AND BMI is " + str(r[5])+" AND DPF is "+str(r[6])
                      + " AND Diabetes Status is " + str(r[7])+"\n")
    return qualified_rules_list

def generateRules():
    f = open("preprocessed_data.csv", "r")
    reader = csv.reader(f)
    rules_list = []
    global values_list
    for row in reader:
        if(len(row) != 0):
            values = [int(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]),
                      float(row[6]), int(row[7])]
            age = fuzzySets(age_low, age_med, age_high, int(row[1]))
            glucose = fuzzySets(glucose_low, glucose_med, glucose_high, float(row[2]))
            dbp = fuzzySets(dbp_low, dbp_med, dbp_high, float(row[3]))
            insulin = fuzzySets(insulin_low, insulin_med, insulin_high, float(row[4]))
            bmi = fuzzySets(bmi_low, bmi_med, bmi_high, float(row[5]))
            dpf = fuzzySets(dpf_low, dpf_med, dpf_high, float(row[6]))
            diab = int(row[7])
            rule = [int(row[0]),age, glucose, dbp, insulin, bmi, dpf, diab]
            rules_list.append(rule)
            values_list.append(values)

    return rules_list, values_list


if __name__ == '__main__':
    rules, values = generateRules()

    qualified_rules = check_conflict(rules)
    with open("test_rules.csv", "w+") as r_f:
        for q in qualified_rules:
            r = values[q[0]]
            r_f.write(str(r[1])+","+str(r[2])+","+str(r[3])+","+str(r[4])+","+str(r[5])+","+str(r[6])+","+str(r[7])+"\n")
            print(r[0])