courses=['history','math','chem','biology']
print(courses)
print(len(courses))
print(courses[-2])
courses.append('art')#extend
print(courses)
courses.insert(1,'phy')
print(courses)
print('math' in courses)
i=1
for item in courses:
	print(str(i)+" "+item)
	i+=1
#tuples cant be modified
tuple_1=(1,2,3,4,5)	
#sets dont care abt order, remove duplicate values
set_1={1,2,3,4,5}
empty_list=list()
empty_set=set()
print(empty_set)
print(empty_list)
#dictionaries key value
student={'name':'sushant','age':19,'courses':['history','math']}
student.update({'age':24})
print(student.get('age'))
for n in tuple_1:
	if n==4:
		print('Found')
		break
	print(n)	
