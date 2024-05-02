add_executable(MemCoalescing MemCoalescingTest.cpp)
target_link_libraries(MemCoalescing gtest_main hip::device)
add_test(NAME MemCoalescingTest COMMAND MemCoalescing)