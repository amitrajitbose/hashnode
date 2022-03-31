## SQL Handbook

In this blog we will just go through the most important and frequently used fundamental features of SQL (the language used to query relational databases such as MySQL, etc)

## [Data Types](https://docs.oracle.com/cd/B19306_01/server.102/b14200/sql_elements001.htm) in Oracle SQL
- VARCHAR2(size)
- NVARCHAR2(size)
- NUMBER[(precision)
- LONG
- DATE
- BINARY_FLOAT
- BINARY_DOUBLE
- CHAR [(size)]
*N.B: This tutorial uses SQLite, although most features are same as Oracle SQL and other variants*

## SQL Manipulation

- Create Table `CREATE TABLE celebs(id INTEGER, name TEXT, age INTEGER );`
- Insert Rows `INSERT INTO celebs (id, name, age) VALUES (1, 'Justin Bieber', 21);`
- Select Queries `SELECT * FROM mytable;`
- Update Value `UPDATE celebs SET age = 22 WHERE id = 1;`
- Alter Table `ALTER TABLE celebs ADD COLUMN twitter_handle TEXT;`
- Delete Rows `DELETE FROM celebs WHERE twitter_handle IS NULL;`
- Add Constraint `CREATE TABLE awards (id INTEGER PRIMARY KEY, recipient TEXT NOT NULL, award_name TEXT DEFAULT "Grammy" );`<br>

```
CREATE TABLE celebs ( 
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,
  date_of_birth TEXT NOT NULL,
  date_of_death TEXT DEFAULT 'Not Applicable'
  );
```
### More on constraints:
Constraints that add information about how a column can be used are invoked after specifying the data type for a column. They can be used to tell the database to reject inserted data that does not adhere to a certain restriction.

1.**PRIMARY KEY** columns can be used to uniquely identify the row. Attempts to insert a row with an identical value to a row already in the table will result in a constraint violation which will not allow you to insert the new row.

2. **UNIQUE** columns have a different value for every row. This is similar to PRIMARY KEY except a table can have many different UNIQUE columns.

3. **NOT NULL** columns must have a value. Attempts to insert a row without a value for a NOT NULL column will result in a constraint violation and the new row will not be inserted.

4. **DEFAULT** columns take an additional argument that will be the assumed value for an inserted row if the new row does not specify a value for that column.



## SQL Queries

- Select Columns `SELECT NAME, GENRE, YEAR FROM MOVIES;`
- Rename Columns/Aliasing `SELECT imdb_rating AS 'IMDb' FROM MOVIES;`
- Select Distinct `SELECT DISTINCT year FROM MOVIES;`
- Where Clause `SELECT * FROM movies WHERE YEAR > 2014;`
- LIKE (Type 1) `SELECT * FROM movies WHERE name LIKE 'Se_en';`
- LIKE (Type 2) `SELECT * FROM movies WHERE name LIKE 'The %';`
- LIKE (Type 3) `SELECT * FROM movies WHERE name LIKE '%N';`
- LIKE (Type 4) `SELECT * FROM movies WHERE name LIKE '%man%';`
- Null/Not Null `SELECT name FROM movies WHERE imdb_rating IS NOT NULL;`
- Between (Type 1 - If number, then both inclusive) `SELECT * FROM movies WHERE YEAR BETWEEN '1970' AND '1979';`
- Between (Type 2 - If char, then both excluding the second number) `SELECT * FROM movies WHERE name BETWEEN 'A' AND 'J';`
- AND Connector `... WHERE year < 1985 AND GENRE='horror';`
- OR Connector `... WHERE genre = 'comedy' OR genre = 'romance';`
- ORDER BY (ascending ASC) `SELECT name FROM movies ORDER BY imdb_rating;`
- ORDER BY (descending DESC) `SELECT name FROM movies ORDER BY imdb_rating DESC;`
- Limit the prompt from showing all rows `SELECT * FROM movies ORDER BY imdb_rating DESC LIMIT 3;`
- Conditional Logic If-Then <br>

```
SELECT name,
 CASE
  WHEN imdb_rating > 8 THEN 'Fantastic'
  WHEN imdb_rating > 6 THEN 'Poorly Received'
  ELSE 'Avoid at All Costs'
 END AS 'Reviews
FROM movies;
```

## Aggregate Functions
Aggregate functions combine multiple rows together to form a single value of more meaningful information.

- Count total rows in a column `SELECT COUNT(*) FROM table_name;`
- Sum of values in a column `SELECT SUM(downloads) FROM fake_apps;`
- Maximum value in a column `SELECT MAX(downloads) FROM fake_apps;`
- Minimum value in a column `SELECT MIN(downloads) FROM fake_apps;`
- Average of a column `SELECT AVG(downloads) FROM fake_apps;`
- Round the values a column to the number of decimal places specified <br>

```
SELECT ROUND(price, 0) FROM fake_apps;
SELECT name, ROUND(AVG(price), 2) FROM fake_apps;
```
- Grouping similar values `SELECT price, COUNT(*) FROM fake_apps GROUP BY price;`
- Grouping values `SELECT ROUND(imdb_rating),COUNT(name) FROM movies GROUP BY 1 ORDER BY 1;` Here 1 refers to the first parameter in the SELECT, i.e ROUND(imdb_rating).
- Having is used to filter Group By results <br>

```
SELECT price,
   ROUND(AVG(downloads)),
   COUNT(*)
FROM fake_apps
GROUP BY price 
HAVING COUNT(*) > 10;

```
GROUP BY is a clause used with aggregate functions to combine data from one or more columns.<br>
HAVING limit the results of a query based on an aggregate property.<br>


## Multiple Tables
In order to efficiently store data, we often spread related information across multiple tables.

- **Joining Multiple Tables** 
```
SELECT orders.order_id,customers.customer_name
FROM orders
JOIN customers
  ON orders.customer_id = customers.customer_id;
```
Let's break down this command:

- The first line selects certain columns, we can specify which ones we want or use * (asterisk).
- The second line specifies the first table that we want to look in, orders
- The third line uses JOIN to say that we want to combine information from orders with customers.
- The fourth line tells us how to combine the two tables. We want to match orders table's customer_id column with customers table's customer_id column.
- Because column names are often repeated across multiple tables, we use the syntax table_name.column_name to be sure that our requests for columns are unambiguous.
- In our example, we use this syntax in the ON statement, but we will also use it in the SELECT or any other statement where we refer to column names.


- **Inner Join**

When we perform a simple JOIN (often called an inner join) our result only includes rows that match our ON condition.
Consider the following animation, which illustrates an inner join of two tables on table1.c2 = table2.c2:

![inner join gif](https://s3.amazonaws.com/codecademy-content/courses/learn-sql/multiple-tables/inner-join.gif)
[Link To GIF](https://s3.amazonaws.com/codecademy-content/courses/learn-sql/multiple-tables/inner-join.gif)

We have seen the query structure above.

- **Left Join**

What if we want to combine two tables and keep some of the un-matched rows?
SQL lets us do this through a command called LEFT JOIN. A left join will keep all rows from the first table, 
regardless of whether there is a matching row in the second table.
Consider the following animation:

![left join gif](https://s3.amazonaws.com/codecademy-content/courses/learn-sql/multiple-tables/left-join.gif)
[Link To GIF](https://s3.amazonaws.com/codecademy-content/courses/learn-sql/multiple-tables/left-join.gif)

The query structure looks like :
```
SELECT *
FROM table1
LEFT JOIN table2
  ON table1.c2 = table2.c2;
```

- **Primary & Foreign Keys**

Primary keys have a few requirements:
- None of the values can be NULL.
- Each value must be unique (i.e., you can't have two customers with the same *customer_id* in the *customers* table).
- A table can not have more than one primary key column.

When the primary key for one table appears in a different table, it is called a **foreign key**.

Why is this important? 
The most common types of joins will be joining a foreign key from one table with the primary key from another table. 
For instance, when we join orders and customers, we join on customer_id, which is a foreign key in orders and the primary key in customers.

- **Cross Join**

Cross Join is like cartesian product. Sometimes, we just want to combine all rows of one table with all rows of another table.
For instance, if we had a table of shirts and a table of pants, we might want to know all the possible combinations to create different outfits.
Cross joins do not require an **ON** clause, because they are not really joining on any condition.

The query structure looks like:
```
SELECT shirts.shirt_color,
   pants.pants_color
FROM shirts
CROSS JOIN pants;
```

This is an example query, where we had a *newspaper* table and a *months* table. This query returns the total number of subscriptions for each month.
```
SELECT month, COUNT(*)
FROM newspaper
CROSS JOIN months
WHERE start_month<=month
AND month<=end_month
GROUP BY month;
```

- **Union**

Basically helps us to stack one table on another. Join was merging side by side. This will stack one on another.
But wait, SQL has strict rules for appending data:
- Tables must have the same number of columns.
- The columns must have the same data types in the same order as the first table.

The query structure looks like:
```
SELECT *
FROM table1
UNION
SELECT *
FROM table2;
```

- **With**

Allows us to define one or more temporary tables that can be used in the final query.
```
WITH previous_results AS (
   SELECT ...
   ...
   ...
   ...
)
SELECT *
FROM previous_results
JOIN customers
  ON _____ = _____;
```
The first query gets an alias **previous_results**, we can now use that table howsoever we want.

Here, is a complex example below.

```
WITH previous_query AS (
SELECT customer_id,
   COUNT(subscription_id) AS 'subscriptions'
FROM orders
GROUP BY customer_id
)
SELECT customers.customer_name, previous_query.subscriptions
FROM previous_query
JOIN customers
ON previous_query.customer_id=customers.customer_id;
```
Essentially what we are doing above is:

- Generate a temporary table and name it as **previous_query**
- The temporary table has a customer ID and total subscriptions made by the customer, thus two columns
- But the customer ID is not good looking, we want the customer name corresponding to it rather
- Thus we join the customer table
- Extract the names where the customer IDs match
- Thus print those corresponding names.

Pheww..! :p

Hope this helps. :)

--------------------------
*Source : [Codecademy](https://www.codecademy.com/learn/learn-sql)*
--------
I certainly believe you learned something out of this blog if you followed it carefully. As a reward for my time and hard work feel free to [buy me a beer or coffee](https://www.buymeacoffee.com/amitrajit).