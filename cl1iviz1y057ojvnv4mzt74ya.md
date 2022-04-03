## Intro To Database Management System

# Introduction To DBMS
This is essentially a blog that came out as a result of an NPTEL course I took during my college days.

### Schema & Instance
- Schema - A schema is a physical or logical structure of the data-base. The basically comprise the columns along with the table.
- Instance - This is the actual data, when the schema is populated by real values and data entries in the form of rows.

### Data Model
- It is the collection of tools that describe the data.
- *Data* : The data itself is an integral part of the data-base.
- *Data Relationships* : Relationship between various data and fields.
- *Data Semantics* : Refers to the meaning of the data.
- *Data Constraints* : Limitations and conditions on certain data values and their modifications.

### Data Model - Types
- ER Model : It does not contain any data. It is only used for designing the data-base.
- Relational Model : Most widely used model for data-base, will be covered in further lessons.
- Object Based Data Models : Object oriented approach of representing data.
- Semi Structured Data Model : Used during data-base transfer, usually in XML format. For example: while transferring from Oracle DB to MySQL DB, etc.
- Network Model
- Hierarchical Model

### DDL & DML
- DDL : *Data Definition Language* : It is the part of the language which is used to define and manipulate the **schema** of the data-base.
- DML : *Data Manipulation Language* : It is the part of the language which is used to query and manipulate the **instance** of the data-base, in other word the data itself. We use SQL for this.

### SQL
- Structured Query Language
- The most widely used commercial language
-It is not a Turing machine equivalent language.
-Usually embedded in some other higher-level language.
-Application programs generally access database through either **language extension** to allow embedded SQL or **APIs like ODBC/JDBC** which allow SQL queries to be sent to the databases.

## Database Design
- Logical design is the first step by business mindset.
- Logical design from computer science mindset.
- Physical design of the schema/layout, the database files, their index, etc.

## Design Approaches
- To minimize redundancies
- Reducing potential anomalies
- We use **ER Model** and/or **Normalization Theory**.

## Object Relational Data Model
- We extend the relational model by using object orientation, constructs and added data types.
- Attributes allowed to have complex types, nested relations, etc.
- Provide upward compatibility with existing relational languages.

## Extensible Markup Language (XML)
- Originally designed as document markup, by W3C.
- Regularly used as the data interchange format, export, import.
- Lot of available tools for XML.

## Database Engine
- Storage manager
- Query processing
- Transaction manager

## Storage Manager
- Storage manager is the bridge between low level data stored in the database and the application programs and queries submitted to the system.
- Responsible for interaction with OS file manager
- Responsible for storing, retrieving and updating data
- Deals with issues like indexing, hashing, file organization, storage access

## Query Processing
- Deals with parsing and translation to relational algebra expressions
- Then it optimizes the queries and expressions
- The evaluation is then done

## Transaction Manager
- Deals issue 1: What if the database/system fails?
- How to recover in case of failure?
- Deals issue 2: What if multiple users are concurrently updating the same data?
- Performs a single logical function in the database application
- Ensures data consistency with the help of **concurrency control manager**

## Users & Admin
- Naive users - end users
- Application Programmers - writes codes and queries
- Sophisticated users/analysts - designs query tools, migrates data, analyse data
- DB Administrators - priviledged access to database, special permissions apart from other permissions

## DB System Internals
- Refer Diagram 1.26

## DB Architectures
- Centralized
- Client Server
- Parallel (Multi Processor)
- Distributed

## History & Resources
- [Course PPT](https://docs.google.com/viewer?a=v&pid=sites&srcid=Z2FyZGl2aWR5YXBpdGguYWMuaW58ZGF0YWJhc2V8Z3g6MmFhNzY1NmQ0ZjUxYjc4ZQ)
- [Optional](http://4840895.blogspot.com/2009/04/history-of-dbms.html)