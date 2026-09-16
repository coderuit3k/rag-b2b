import { SignIn } from "@clerk/nextjs";

export default function Page() {
  return (
    <div className="flex h-dvh items-center justify-center bg-surface">
      <SignIn />
    </div>
  );
}
