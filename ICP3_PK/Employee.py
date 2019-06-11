# Defining the class Employee
class Employee:
    empCount = 0  # variable assigned (data member created) to count the employees
    totalSalary = 0

# Constructor for Employee defined with the variables initialized viz. name, family, salary and department of the employee
    def __init__(self, name, family, salary, dept):
        self.name = name
        self.family = family
        self.salary = salary
        self.dept = dept
        self.__class__.empCount += 1
        self.__class__.totalSalary = Employee.totalSalary + salary

# Create function to call average salary
    def avg_salary(self):
        avgSalary = Employee.totalSalary/Employee.empCount
        return avgSalary

# Create instances (passing arguments) of the employee and calling their member functions
emp1 = Employee('Poonam', 'Kankariya', 7000, 'Finance')
emp2 = Employee('Priyanka', 'Gaikwad', 6000, 'HR')
emp3 = Employee('Rohit', 'Abhishek', 5000, 'Operations')

print(emp1.avg_salary())

# Defining sub class of full time employee inheriting parent class functions
class FulltimeEmp(Employee):
    FTEcount = 0
    FTEtotalSalary = 0
    def __init__(self, name, family, salary, dept):
        Employee.__init__(self, name, family, salary, dept)
        self.name = name
        self.__class__.salary = salary
        self.__class__.FTEcount += 1
        self.__class__.FTEtotalSalary = FulltimeEmp.FTEtotalSalary + salary

# Creating a member function associated with full time employees class
    def fulltimeCount(self):
        fulltimeCount = FulltimeEmp.FTEcount
        return fulltimeCount

# Create function to call average salary
    def avgFTEsalary(self):
        avgFTEsalary = FulltimeEmp.FTEtotalSalary/FulltimeEmp.FTEcount
        return avgFTEsalary

# Create instances (passing arguments) of the full time employee and calling their member functions
emp4 = FulltimeEmp('Anurag', 'Thantharate', 7000, 'Marketing')
emp5 = FulltimeEmp('Vijay', 'Walunj', 7000, 'IT Support')

print(emp4.avgFTEsalary())

# Calling functions for employees and full-time employees
avgSalary = [emp1, emp2, emp3, emp4, emp5]
AverageSal = Employee.avg_salary(avgSalary)
print("Average salary of the employees is $", AverageSal)

# Calling full time employee function to identify the number of employees
totalEmp = emp5.fulltimeCount() + emp1.empCount
print("Total no. of employees in the company are ", totalEmp)

