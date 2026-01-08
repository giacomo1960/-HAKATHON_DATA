# Use a base image with both Java and Python
# We can use a JDK image and install Python, or a Python image and install JDK.
# Using Eclipse Temurin (Java) as base.
FROM eclipse-temurin:17-jdk-jammy

# Install Python and pip, and Maven (added)
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv maven

# Set working directory
WORKDIR /app

# Copy DataScience files and install requirements
COPY DataScience /app/DataScience
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install -r /app/DataScience/requirements.txt

# Copy Backend files
COPY backend /app/backend

# Build the Backend
WORKDIR /app/backend
# Use mvn instead of ./mvnw
RUN mvn clean package -DskipTests

# Expose port
EXPOSE 8080

# Run the application
# We need to make sure the java app runs from a place where it can find DataScience/api.py
# The code looks for "../DataScience/api.py" or "DataScience/api.py"
# If we run the jar from /app/backend/target, and DataScience is at /app/DataScience...
# Let's see.
# If we run: java -jar target/app.jar from /app/backend
# The code does: new File("../DataScience/api.py") -> /app/DataScience/api.py. This should work.

CMD ["java", "-jar", "target/backend-0.0.1-SNAPSHOT.jar"]
