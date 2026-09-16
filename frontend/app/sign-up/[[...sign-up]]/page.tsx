import { SignUp } from "@clerk/nextjs";

export default function Page() {
  return (
    <div className="flex h-dvh items-center justify-center bg-zinc-50">
      <SignUp />
    </div>
  );
}
