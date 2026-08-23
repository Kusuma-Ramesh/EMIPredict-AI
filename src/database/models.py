from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from database.database import Base


class Customer(Base):

    __tablename__ = "customers"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    age = Column(Float, nullable=False)
    gender = Column(String, nullable=False)
    marital_status = Column(String, nullable=False)
    education = Column(String, nullable=False)

    monthly_salary = Column(Float, nullable=False)

    employment_type = Column(String, nullable=False)
    years_of_employment = Column(Float, nullable=False)
    company_type = Column(String, nullable=False)
    house_type = Column(String, nullable=False)

    monthly_rent = Column(Float, nullable=False)
    family_size = Column(Integer, nullable=False)
    dependents = Column(Integer, nullable=False)

    school_fees = Column(Float, nullable=False)
    college_fees = Column(Float, nullable=False)
    travel_expenses = Column(Float, nullable=False)
    groceries_utilities = Column(Float, nullable=False)
    other_monthly_expenses = Column(Float, nullable=False)

    existing_loans = Column(String, nullable=False)
    current_emi_amount = Column(Float, nullable=False)

    credit_score = Column(Float, nullable=False)
    bank_balance = Column(Float, nullable=False)
    emergency_fund = Column(Float, nullable=False)

    emi_scenario = Column(String, nullable=False)
    requested_amount = Column(Float, nullable=False)
    requested_tenure = Column(Integer, nullable=False)

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    customer_id = Column(
        Integer,
        nullable=True
    )

    emi_status = Column(
        String,
        nullable=False
    )

    not_eligible_probability = Column(
        Float,
        nullable=True
    )

    eligible_probability = Column(
        Float,
        nullable=True
    )

    high_risk_probability = Column(
        Float,
        nullable=True
    )

    predicted_max_monthly_emi = Column(
        Float,
        nullable=True
    )

    requested_monthly_emi = Column(
        Float,
        nullable=True
    )

    recommendation = Column(
        String,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
