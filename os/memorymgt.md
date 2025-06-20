Reference
https://pages.cs.wisc.edu/~remzi/Classes/537/Spring2018/Book/vm-tlbs.pdf

Notes
How do we make sure paging is faster, as each virtual memory translation needs memory access for vitrual to physical address translation?

TLB - Translation Look Aside Buffer - part of chip (hardware) MMU memory management unit

A cache in chip for PTE

8-bit address space, 2^8 total addressable memory for a process = 256 addresses, with 16-byte page, how many pages a program can have - 2^8/2^4 = 2^4 = 16 pages = 4 bit for VPN in a address space

TLB Content / per process wise - VPN | PFN (process frame number) | valid bits

TLB Context Switch

Either OS execute a priviledge instruction to flush TLS at context switich
Or Hardware marks valid bits as 0 so that new process PTE can be cached as the new process executes
But both options are costly
One idea TLB can share PTEs for multiple processes
Some hardware provides Address Space Identifier ASID a process identifier in TLB . OS must set a special register with new ASID when context swtich happens
TLB Cache Replacement

LRU (may be)
Random Policy - Evict Random pages, simple and works well in edge cases