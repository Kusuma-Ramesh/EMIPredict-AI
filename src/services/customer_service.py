from sqlalchemy.orm import Session

from database.models import Customer
from schemas.customer import CustomerCreate, CustomerUpdate


def create_customer(
    db: Session,
    customer_data: CustomerCreate
):
    customer = Customer(
        **customer_data.model_dump()
    )

    db.add(customer)
    db.commit()
    db.refresh(customer)

    return customer


def get_customers(
    db: Session,
    skip: int = 0,
    limit: int = 100
):
    return (
        db.query(Customer)
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_customer(
    db: Session,
    customer_id: int
):
    return (
        db.query(Customer)
        .filter(
            Customer.id == customer_id
        )
        .first()
    )


def update_customer(
    db: Session,
    customer,
    customer_data: CustomerUpdate
):
    update_data = (
        customer_data
        .model_dump(exclude_unset=True)
    )

    for field, value in update_data.items():
        setattr(
            customer,
            field,
            value
        )

    db.commit()
    db.refresh(customer)

    return customer


def delete_customer(
    db: Session,
    customer
):
    db.delete(customer)
    db.commit()
