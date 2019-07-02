# Class 1: Flight which contains the flight number(f_id), its origin and destination, the number of stops between the
#  origin and destination and the type of airlines(f_type)
class Flight():
    # INIT CONSTRUCTOR
    def _init_(self, f_id, f_origin, f_destination, no_of_stops, flight_type, p_id, p_type):
        self.f_id = f_id
        self.origin = f_origin
        self.destination = f_destination
        self.stops = no_of_stops
        self.flight_type = flight_type
        self.pid = p_id
        self.ptype = p_type

    def get_flight_details(self, f_id):
        print("Flight No:", f_id)
        print("ORG:", self.origin)
        print("DEST:", self.destination)
        print("Flight Type:", self.flight_type)

# Class2: Person which contains the personID(p_id), their name, phone number, gender, type of person(
# employee/passenger) and it inherits the Flight class to get the flight details.

class Person(Flight):
    # INIT CONSTRUCTOR
    def _init_(self, p_id, p_type, p_gender, p_name, p_phonenumber, f_id, f_origin, f_destination, no_of_stops, flight_type):
        self.name = p_name
        self.gender = p_gender
        self.p_phonenumber = p_phonenumber
# Here we also use super class to use the parameters from Flight class
        super(Person, self)._init_(f_id, f_origin, f_destination, no_of_stops, flight_type, p_id, p_type)

# Here we used MULTIPLE INHERITANCE as the Person is derived from Flight and the Employee and Passenger is derived
# from Person.

# Class3: Employee which is an inherited class from Person, SSN is the private data member, since we cant reveal the
# SSN.

class Employee(Person):
    # INIT CONSTRUCTOR
    def _init_(self, p_id, p_type, p_gender, p_name, p_phonenumber, f_id, e_SSN, f_origin, f_destination, no_of_stops, flight_type):
        super(Employee,self)._init_(p_id, p_type, p_gender, p_name, p_phonenumber, f_id, f_origin, f_destination, no_of_stops, flight_type)
        self.__emp_SSN = e_SSN

# This method is to get the travel details of the employee
    def get_travel_details_employee(self):
       # print("Travel Details of ", self.emp_SSN)
        print("Hello Pilot ", self.name, "Here are your flight details")
        print("Flight_ID:", self.f_id)
        print("ORG:", self.origin)
        print("DEST:", self.destination)

# Class 4:Passenger which is an inherited class from Person, Passport Number is the private data member,
# since we cant reveal it.
class Passenger(Person):
    names = []
    d = dict()
    # INIT CONSTRUCTOR

    def _init_(self, p_id, p_type, p_gender, p_name, p_phonenumber, f_id, pno, f_origin, f_destination, no_of_stops, flight_type):
        super(Passenger, self)._init_(p_id, p_type, p_gender, p_name, p_phonenumber, f_id, f_origin, f_destination, no_of_stops, flight_type)
        self.pno = pno

# This is to get the travellers on the plane into a list, where we have the flightNumber(f_id)
        # as the key and the passengername(name) as the value.
        if self.f_id in Passenger.d.keys():
            Passenger.d[self.f_id].append(self.name)
        else:
            Passenger.d[self.f_id] = [self.name]

    # This method is to get the travel details of the passenger
    def get_travel_details_passanger(self):
        print("Travel Details of ", self.name)
        print("Flight Id:",   self.f_id)
        print("Flight Type:", self.flight_type)
        print("ORG:", self.origin)
        print("DEST:", self.destination)

# This method is to print the dictionary where we have stored the passengers list for different flights
    def get_travelling_passengers(self):
        print("Passengers on the flight", Passenger.d)


class Ticket(Passenger):
    def _init_(self, p_id, p_type, p_gender, p_name, p_phonenumber, f_id, pno, f_origin, f_destination, no_of_stops,
                 flight_type, boarding_group_no, row, seat_no):
        super(Ticket, self)._init_(p_id, p_type, p_gender, p_name, p_phonenumber, f_id, pno, f_origin, f_destination,
                                     no_of_stops, flight_type)
        self.boarding_group_no = boarding_group_no
        self.row = row
        self.seat_no = seat_no
        print("Your ticket details are below: ")

    def get_boarding_pass(self, p_name):
        for k, v in Passenger.d.items():
            names = v
            for i in names:
                if i == p_name:
                    print("Passenger Name:", p_name)
                    print("Flight Id:", k)
                    print("Boarding Group and Seat No:", self.boarding_group_no, self.row, self.seat_no)
                    print("ORG:", self.origin)
                    print("DEST:", self.destination)

# p_id,p_type,p_gender,p_name,p_phonenumber,f_id,pno,f_origin,f_destination,no_of_stops,flight_type
#f_id, f_origin, f_destination, no_of_stops, flight_type, p_id, p_type
f1 = Flight("AE1425", "PHL", "MCI", 1, "Air India", "P1", "P4")
f1.get_flight_details("AE1425")

#Instances of Passenger class
p1 = Passenger("P1", "P", "F", "NIKKI", 4258796358, "F1400", "A123", "MCI", "HYD", 3, "Air India")
p2 = Passenger("P2", "P", "M", "ANUSHA", 2588796358, "F1445", "A123", "MCI", "HYD", 3, "Air India")
p3 = Passenger("P3", "P", "F", "SRAVANTHI", 42587961558, "F1340", "A123", "MCI", "BLR", 3, "QATAR")

#Instances of Employee class
e1 = Employee("E1", "E", "M", "ASHISH", 142587961558, "F1578", "A123", "MCI", "HYD", 3, "Air India")
e2 = Employee("E2", "E", "M", "AKASH", 422587961558, "F1234", "A123", "MCI", "HYD", 3, "Air India")
e3 = Employee("E2", "E", "M", "JOHN", 424587961558, "F1647", "A123", "MCI", "BLR", 3, "QATAR")

#This method prints the travel details for the passenger
p1.get_travel_details_passanger()
#This method prints the travel details for the employee
e1.get_travel_details_employee()

#This method prints the travelling passengers on that flight
p1.get_travelling_passengers()

#Prints the boarding pass
T1 = Ticket("P1", "P", "F", "SAI", 4258796358, "F1425", "A123", "MCI", "HYD", 3, "Air India", "G", "E", 12)
T1.get_boarding_pass("SAI")