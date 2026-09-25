-- Q1: SELECT / WHERE
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp > 20
ORDER BY price_gbp DESC;

-- Q2: ORDER BY / LIMIT
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;

-- Q3: DISTINCT
SELECT DISTINCT rating
FROM books
ORDER BY rating;

-- Q4: IN
SELECT title, price_gbp, category_id
FROM books
WHERE rating IN (4, 5)
ORDER BY price_gbp DESC;

-- Q5: BETWEEN
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 10 AND 30
ORDER BY price_gbp;

-- Q6: JOIN
SELECT b.title, b.price_gbp, b.rating, c.name AS category
FROM books AS b
JOIN categories AS c ON b.category_id = c.category_id
ORDER BY b.price_gbp DESC
LIMIT 15;
