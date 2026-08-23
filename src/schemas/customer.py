from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CustomerBase(BaseModel):

    age: float = Field(..., ge=18, le=100)
    gender: str
    marital_status: str
    education: str

    monthly_salary: float = Field(..., gt=0)

    employment_type: str
    years_of_employment: float = Field(..., ge=0)
    company_type: str
    house_type: str

    monthly_rent: float = Field(..., ge=0)
    family_size: int = Field(..., ge=1)
    dependents: int = Field(..., ge=0)

    school_fees: float = Field(..., ge=0)
    college_fees: float = Field(..., ge=0)
    travel_expenses: float = Field(..., ge=0)
    groceries_utilities: float = Field(..., ge=0)
    other_monthly_expenses: float = Field(..., ge=0)

    existing_loans: str
    current_emi_amount: float = Field(..., ge=0)

    credit_score: float = Field(..., ge=300, le=850)
    bank_balance: float = Field(..., ge=0)
    emergency_fund: float = Field(..., ge=0)

    emi_scenario: str
    requested_amount: float = Field(..., gt=0)
    requested_tenure: int = Field(..., ge=1)


class CustomerCreate(CustomerBase):
    pass


class CustomerUpdate(BaseModel):

    age: float | None = Field(None, ge=18, le=100)
    gender: str | None = None
    marital_status: str | None = None
    education: str | None = None

    monthly_salary: float | None = Field(None, gt=0)

    employment_type: str | None = None
    years_of_employment: float | None = Field(None, ge=0)
    company_type: str | None = None
    house_type: str | None = None

    monthly_rent: float | None = Field(None, ge=0)
    family_size: int | None = Field(None, ge=1)
    dependents: int | None = Field(None, ge=0)

    school_fees: float | None = Field(None, ge=0)
    college_fees: float | None = Field(None, ge=0)
    travel_expenses: float | None = Field(None, ge=0)
    groceries_utilities: float | None = Field(None, ge=0)
    other_monthly_expenses: float | None = Field(None, ge=0)

    existing_loans: str | None = None
    current_emi_amount: float | None = Field(None, ge=0)

    credit_score: float | None = Field(None, ge=300, le=850)
    bank_balance: float | None = Field(None, ge=0)
    emergency_fund: float | None = Field(None, ge=0)

    emi_scenario: str | None = None
    requested_amount: float | None = Field(None, gt=0)
    requested_tenure: int | None = Field(None, ge=1)


class CustomerResponse(CustomerBase):

    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )
