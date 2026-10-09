"""Gold data for the policy RAG. Each entry lists every clause that is a correct answer.

Questions are phrased the way a finance user would ask, not copied from the clause text.
"""
QUESTIONS = [
    ("How often do we have to reconcile the bank accounts and who signs off?", {"BR-1"}),
    ("A bank line has no reference number. Can I still tick it against the ledger?", {"BR-2"}),
    ("The bank shows one transfer but we booked three vendor payments. Is that a valid match?", {"BR-3"}),
    ("How do I account for the fees the bank deducted this month?", {"BR-4"}),
    ("The bank paid us interest on our deposit. What is the journal entry?", {"BR-5"}),
    ("Money came into the account and nobody knows which customer sent it. What now?", {"BR-6"}),
    ("There is a withdrawal on the statement that we never recorded. Who do I tell?", {"BR-7"}),
    ("A cheque we issued four months ago was never cashed. What should we do?", {"BR-8"}),
    ("Can accounts book a supplier bill that has no PO?", {"AP-1", "MSA-3"}),
    ("The goods have not arrived yet. Can we pay the supplier anyway?", {"AP-2"}),
    ("The supplier billed 6% more than the agreed rate. Is that allowed?", {"AP-3", "MSA-1"}),
    ("We ordered 100 units and got 80. How much do we pay?", {"AP-4", "MSA-7"}),
    ("The same bill seems to have been entered twice. What is the procedure?", {"AP-5", "MSA-3"}),
    ("The tax on a bill does not add up to the rate times the value. Do we accept it?", {"TAX-1"}),
    ("When must the tax we withheld from vendors be deposited?", {"TAX-2"}),
    ("When are we allowed to claim input credit on a purchase?", {"TAX-3", "TAX-1"}),
    ("Up to what amount can an AP executive sign off a bill alone?", {"DOA-1"}),
    ("A vendor sent two bills on one day, each just below the CFO threshold. Is that a problem?", {"DOA-2"}),
    ("Can the employee who set up a supplier also release that supplier's payments?", {"DOA-3"}),
    ("Someone booked entries on a Sunday. Is that permitted?", {"DOA-4"}),
    ("Why do we question a bill for exactly 50,000?", {"DOA-5"}),
    ("What documents do we need before adding a new supplier?", {"VM-1"}),
    ("A supplier emailed new account details. How do we verify the request?", {"VM-2", "MSA-5"}),
    ("How many days does the supplier give us to settle a bill?", {"MSA-4"}),
    ("Who owns the goods once they reach our warehouse, and can we send back damaged ones?", {"MSA-2"}),
    ("What penalty applies if the supplier delivers late?", {"MSA-7"}),
]

# Correct clause(s) for each approval-queue finding kind the deterministic agents produce.
KIND_CLAUSE = {
    "bank_charge": {"BR-4"}, "interest": {"BR-5"}, "unrecorded_receipt": {"BR-6"}, "unrecorded_payment": {"BR-7"},
    "outstanding_item": {"BR-8"},
    "no_po": {"AP-1"}, "price_mismatch": {"AP-3", "MSA-1"}, "no_grn": {"AP-2"}, "short_receipt": {"AP-4", "MSA-7"},
    "gst_error": {"TAX-1"}, "duplicate_invoice": {"AP-5"},
    "split_invoice": {"DOA-2"}, "sod_breach": {"DOA-3"}, "weekend_posting": {"DOA-4"}, "round_amount": {"DOA-5"},
    "bank_change": {"VM-2", "MSA-5"},
}
