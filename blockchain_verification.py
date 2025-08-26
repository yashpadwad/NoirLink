"""
Improved blockchain_verification.py
----------------------------------
- Deploys IntegrityStore contract on Ganache
- Computes SHA256 hash of clustered CSV
- Stores hash on-chain
- Saves contract ABI + address for Streamlit dashboard
"""

import json
import os
import hashlib
from web3 import Web3
from solcx import compile_source, install_solc

# -----------------------------
# Solidity contract
# -----------------------------
CONTRACT_SOURCE = r'''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract IntegrityStore {
    bytes32 public storedHash;
    address public owner;

    event HashStored(bytes32 indexed newHash, address indexed by);

    constructor() {
        owner = msg.sender;
    }

    function storeHash(bytes32 h) public {
        storedHash = h;
        emit HashStored(h, msg.sender);
    }
}
'''

# -----------------------------
# Paths
# -----------------------------
CSV_PATH = "data/cdr_clusters.csv"
GANACHE_RPC = "http://127.0.0.1:7545"
ABI_FILE = "contract_abi.json"
ADDR_FILE = "contract_address.txt"

# -----------------------------
# Helpers
# -----------------------------
def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            h.update(chunk)
    return "0x" + h.hexdigest()

# -----------------------------
# Main flow
# -----------------------------
def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run idpc_encrypted.py first.")
        return

    file_hash = sha256_of_file(CSV_PATH)
    print("Local CSV SHA256:", file_hash)

    # Connect to Ganache
    w3 = Web3(Web3.HTTPProvider(GANACHE_RPC))
    if not w3.is_connected():
        print("ERROR: Could not connect to Ganache at", GANACHE_RPC)
        return
    print("Connected to Ganache OK.")

    acct = w3.eth.accounts[0]
    print("Using Ganache account:", acct)

    # Compile contract
    try:
        install_solc("0.8.17")
    except Exception:
        pass
    compiled = compile_source(CONTRACT_SOURCE, solc_version="0.8.17")
    _, contract_interface = compiled.popitem()
    bytecode = contract_interface["bin"]
    abi = contract_interface["abi"]

    # Deploy
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    print("Deploying contract...")
    tx_hash = Contract.constructor().transact({"from": acct})
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract_address = tx_receipt.contractAddress
    print("Contract deployed at:", contract_address)

    # Interact
    contract = w3.eth.contract(address=contract_address, abi=abi)
    print("Storing file hash on-chain...")
    tx = contract.functions.storeHash(file_hash).transact({"from": acct})
    w3.eth.wait_for_transaction_receipt(tx)
    print("Stored on-chain. Tx hash:", tx.hex())

    # Save ABI + Address for Streamlit
    with open(ABI_FILE, "w") as f:
        json.dump(abi, f)
    with open(ADDR_FILE, "w") as f:
        f.write(contract_address)

    print(f"✅ ABI saved to {ABI_FILE}, address saved to {ADDR_FILE}")

    # Verification
    onchain_hash = contract.functions.storedHash().call()
    onchain_hex = w3.to_hex(onchain_hash)

    print("On-chain stored hash:", onchain_hex)
    print("Local CSV hash      :", file_hash)

    if onchain_hex.lower() == file_hash.lower():
        print("✅ Verification SUCCESS: on-chain hash matches local CSV hash.")
    else:
        print("❌ Verification FAILURE: mismatch between on-chain and local hash.")

if __name__ == "__main__":
    main()
