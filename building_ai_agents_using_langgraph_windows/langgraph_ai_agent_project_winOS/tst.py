import asyncio
import time

async def serve_customer(name):
    print(f"[{name}] Starting Order...")

    await asyncio.sleep(2)
    print(f"[{name}] Order placed.")

    await asyncio.sleep(5)
    print(f"[{name}] Payment completed")

async def main_async():
    start_time = time.time()
    print("start_time is: ", start_time)

    await asyncio.gather(serve_customer("Customer_A"), serve_customer("Customer_B"),
    serve_customer("Customer_C"), serve_customer("Customer_D"),
    serve_customer("Customer_E"), serve_customer("Customer_F"),
    serve_customer("Customer_G"), serve_customer("Customer_H"))

    end_time = time.time()

    print("Total Execution time is: ", end_time - start_time)



asyncio.run(main_async())