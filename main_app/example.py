detections[0][1][0].attributes
allAttributes = detections[0][1]
# [Result1, Result2, Result3 , ... ]
# Result1.attributes = {has_hat: 0.5, is_male: 0.1, ...}
peopleWithHelmets = 0

numOfPeople = allAttributes.length()
for person in allAttributes:
    if person.attributes.has_hat > 0.5:
        peopleWithHelmets += 1
    
if peopleWithHelmets < numOfPeople:
    print("WARINING PEOPLE WITHOUT HELMET")
