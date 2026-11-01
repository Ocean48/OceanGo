"""
Backwards-compatibility forwarder for OceanGo.
GPU.py has been renamed to OceanGo.py.
"""
import sys
import os

if __name__ == "__main__":
    print("[Notice] GPU.py has been renamed to OceanGo.py. Forwarding execution...\n")
    from OceanGo import main
    main()
